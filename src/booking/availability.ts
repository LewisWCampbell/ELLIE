import { config } from "../config";
import { BusyRange, CalendarClient } from "./calendar";
import { parseStart, sharedCalendar } from "./booking";

/**
 * The `check_availability` tool: the read half of booking.
 *
 * Availability used to be checked only inside `book_intro_call`, which made the
 * read and the write one indivisible step. That forced Elle to either promise a
 * time before knowing it was open, or collect a name and an email and read the
 * slot back before discovering it was taken. Splitting the read out lets her
 * answer "is Thursday at two free?" on its own, offer real alternatives when it
 * is not, and only then gather what a booking needs.
 *
 * It also separates two very different kinds of call. A read has no side
 * effects, so it can be retried, cached or run early without harm; a write
 * cannot. `book_intro_call` still re-checks the slot at write time, because
 * between this read and that write someone else can take it.
 *
 * Results always start with FREE:, BUSY: or UNKNOWN:, so the model (and the
 * training data built from it) has one unambiguous token to condition on.
 */

const ownerName = config.ownerName;

export const CHECK_AVAILABILITY = {
  name: "check_availability",
  description:
    `Check whether ${ownerName} is free for a ${config.booking.durationMinutes}-minute intro call ` +
    `starting at a specific time. Call this BEFORE telling the caller whether a time works, and ` +
    `before collecting their details for a booking. Never say a time is free or busy until this ` +
    `returns; while it runs, say something brief like "let me check his calendar". If the time ` +
    `is busy, the result lists nearby open times to offer instead. Give 'start' as ISO 8601 with ` +
    `the correct timezone offset (see the current time in context).`,
  parameters: {
    type: "object",
    properties: {
      start: {
        type: "string",
        description: "Proposed start, ISO 8601 with offset, e.g. 2026-08-25T14:00:00-07:00.",
      },
    },
    required: ["start"],
  },
} as const;

const SLOT_MS = 30 * 60_000;
const SEARCH_DAYS = 3;
const MAX_ALTERNATIVES = 3;
const MAX_LEAD_DAYS = 365;

/** Weekday (0 = Sunday) and fractional hour of an instant, in the owner's zone. */
function localClock(d: Date, timeZone: string): { weekday: number; hour: number } {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(d);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  const weekday = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].indexOf(get("weekday"));
  return { weekday, hour: Number(get("hour")) + Number(get("minute")) / 60 };
}

/** "Thursday, November 12 at 2:00 PM PST": how Elle should say a time out loud. */
export function speakableTime(d: Date, timeZone = config.booking.timezone): string {
  return d.toLocaleString("en-US", {
    timeZone,
    weekday: "long",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).replace(/, (\d{1,2}:\d{2})/, " at $1");
}

function overlaps(startMs: number, endMs: number, busy: BusyRange[]): boolean {
  return busy.some((b) => Date.parse(b.start) < endMs && Date.parse(b.end) > startMs);
}

/**
 * Open slots near the requested one, inside working hours, nearest first and
 * then put back in time order so they read naturally. Working hours shape the
 * suggestions only: a caller can still ask for, and book, any time.
 */
export function nearbyOpenings(
  requested: Date,
  busy: BusyRange[],
  now: Date,
  opts = {
    timeZone: config.booking.timezone,
    durationMs: config.booking.durationMinutes * 60_000,
    hours: config.booking.hours,
  }
): Date[] {
  const dayStart = new Date(requested.getTime() - (requested.getTime() % SLOT_MS) - 12 * 3_600_000);
  const candidates: Date[] = [];
  for (let t = dayStart.getTime(); t < requested.getTime() + SEARCH_DAYS * 86_400_000; t += SLOT_MS) {
    if (t <= now.getTime() || t === requested.getTime()) continue;
    const { weekday, hour } = localClock(new Date(t), opts.timeZone);
    const endHour = hour + opts.durationMs / 3_600_000;
    if (weekday === 0 || weekday === 6) continue;
    if (hour < opts.hours.start || endHour > opts.hours.end) continue;
    if (overlaps(t, t + opts.durationMs, busy)) continue;
    candidates.push(new Date(t));
  }
  return candidates
    .sort((a, b) => Math.abs(a.getTime() - requested.getTime()) - Math.abs(b.getTime() - requested.getTime()))
    .slice(0, MAX_ALTERNATIVES)
    .sort((a, b) => a.getTime() - b.getTime());
}

const UNKNOWN_ADVICE =
  `Don't tell the caller whether the time is free. Offer to book it as a request pending ` +
  `${ownerName}'s confirmation, or to try again in a moment.`;

/**
 * Answer one availability question. Never throws: every failure becomes an
 * UNKNOWN: result that tells Elle exactly what she may and may not say, because
 * a read that errors must never be spoken as "yes, that's free".
 */
export async function handleCheckAvailability(
  args: Record<string, unknown>,
  cal: CalendarClient | null = sharedCalendar(),
  now: Date = new Date()
): Promise<string> {
  const raw = String(args.start ?? "").trim();
  if (!raw) return "Error: ask the caller for a specific day and time, then check it.";
  const start = parseStart(raw, config.booking.timezone);
  if (!start) return "Error: couldn't read that time. Ask the caller for a specific day and time.";
  if (start.getTime() < now.getTime() - 60_000) {
    return "Error: that time is in the past. Check the date with the caller (the current time is in your context).";
  }
  if (start.getTime() > now.getTime() + MAX_LEAD_DAYS * 86_400_000) {
    return "Error: that time is more than a year out, which usually means the year is wrong. Confirm the date.";
  }
  if (!cal) {
    return `UNKNOWN: no calendar is connected, so availability can't be checked. ${UNKNOWN_ADVICE}`;
  }

  const durationMs = config.booking.durationMinutes * 60_000;
  const end = new Date(start.getTime() + durationMs);
  const said = speakableTime(start);

  try {
    if (!cal.busy) {
      // A client without busy ranges can still answer yes or no, just without
      // suggestions.
      return (await cal.isFree(start.toISOString(), end.toISOString()))
        ? `FREE: ${said} is open. Offer it to the caller.`
        : `BUSY: ${said} is taken. Ask the caller for another time and check that.`;
    }
    const windowStart = new Date(start.getTime() - 12 * 3_600_000);
    const windowEnd = new Date(start.getTime() + SEARCH_DAYS * 86_400_000);
    const busy = await cal.busy(windowStart.toISOString(), windowEnd.toISOString());
    if (!overlaps(start.getTime(), end.getTime(), busy)) {
      return `FREE: ${said} is open. Offer it to the caller.`;
    }
    const alts = nearbyOpenings(start, busy, now);
    return alts.length
      ? `BUSY: ${said} is taken. Nearby open times: ${alts.map((d) => speakableTime(d)).join("; ")}. ` +
          `Offer one or two of these, not the whole list.`
      : `BUSY: ${said} is taken, and nothing nearby in working hours is open. Ask for another day.`;
  } catch (e) {
    console.error("check_availability failed:", e);
    return `UNKNOWN: couldn't read the calendar just now. ${UNKNOWN_ADVICE}`;
  }
}
