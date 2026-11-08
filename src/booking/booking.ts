import { config } from "../config";
import * as db from "../db";
import { CalendarClient, createCalendarClient } from "./calendar";
import { notifyBooking } from "../notify";

/**
 * The `book_intro_call` tool: Elle's closing move. Once a caller agrees to meet,
 * she books a TENTATIVE hold on the owner's Google Calendar (the owner confirms
 * it) and always tells the caller it's pending confirmation. With no calendar
 * configured it still records the request as a lead, so the goal degrades
 * gracefully. The channels (phone + web) share this one handler.
 */

const ownerName = config.ownerName;

export const BOOK_INTRO_CALL = {
  name: "book_intro_call",
  description:
    `Tentatively book a short intro call between the caller and ${ownerName}. Call this once the ` +
    `caller has agreed to meet and you have their name, email, and a specific proposed start time ` +
    `that you have read back to them and they confirmed. Check the time with check_availability ` +
    `first; this tool re-checks it at write time in case someone took it since. The slot is created as TENTATIVE and is ` +
    `NOT confirmed until ${ownerName} approves it — always ` +
    `tell the caller it is pending ${ownerName}'s confirmation and that he will confirm by email. ` +
    `Give 'start' as ISO 8601 with the correct timezone offset (see the current time in context).`,
  parameters: {
    type: "object",
    properties: {
      name: { type: "string", description: "Caller's full name." },
      email: { type: "string", description: "Caller's email; used to send the invite once confirmed." },
      start: {
        type: "string",
        description: "Proposed start, ISO 8601 with offset, e.g. 2026-08-25T14:00:00-07:00.",
      },
      topic: { type: "string", description: "What the intro call is about (optional)." },
    },
    required: ["name", "email", "start"],
  },
} as const;

export interface BookingContext {
  fromNumber?: string | null;
  callSid?: string | null;
  /** Optional hook to surface the booking on a paired browser. */
  notify?: (summary: string) => void;
}

// One calendar client per process; null when booking isn't configured.
const defaultCalendar: CalendarClient | null = createCalendarClient();

/** The process-wide calendar client (null when booking isn't configured). */
export function sharedCalendar(): CalendarClient | null {
  return defaultCalendar;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** An ISO string that carries its own zone: trailing Z, or +HH:MM / -HHMM. */
const HAS_OFFSET_RE = /(Z|[+-]\d{2}:?\d{2})$/i;

/** How far ahead of UTC `timeZone` is at a given instant, in milliseconds. */
function zoneOffsetMs(instant: Date, timeZone: string): number {
  const dtf = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const p: Record<string, string> = {};
  for (const part of dtf.formatToParts(instant)) p[part.type] = part.value;
  // hour can come back as "24" for midnight under hour12: false.
  const asIfUtc = Date.UTC(
    Number(p.year),
    Number(p.month) - 1,
    Number(p.day),
    Number(p.hour) % 24,
    Number(p.minute),
    Number(p.second)
  );
  return asIfUtc - instant.getTime();
}

/**
 * Turn the model's proposed start into an instant.
 *
 * The tool asks for an offset and the persona supplies the owner's timezone, but
 * models drop the offset sometimes, and a bare "2026-08-28T14:00:00" is parsed
 * by JavaScript as LOCAL time. On Cloud Run local time is UTC, so a caller who
 * asked for two in the afternoon was being booked for seven in the morning their
 * time, silently, after Elle had read "two o'clock" back to them and they had
 * agreed. Nobody finds that out until the meeting is missed.
 *
 * So: a string that carries its own offset is trusted as-is; one that does not
 * is read as wall-clock time in the owner's timezone, which is what the caller
 * and the persona both meant.
 */
export function parseStart(raw: string, timeZone: string): Date | null {
  const s = raw.trim().replace(" ", "T");
  if (!s) return null;

  if (HAS_OFFSET_RE.test(s)) {
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const naive = new Date(s + "Z");
  if (Number.isNaN(naive.getTime())) return null;
  // Two passes: the zone's offset has to be evaluated at the resulting instant,
  // and near a daylight-saving boundary the first guess can land on the wrong
  // side of the change.
  const first = new Date(naive.getTime() - zoneOffsetMs(naive, timeZone));
  return new Date(naive.getTime() - zoneOffsetMs(first, timeZone));
}

/** A booking must be ahead of now, and not absurdly far ahead. */
const MAX_LEAD_DAYS = 365;

/**
 * Execute a booking. Returns the tool-result string handed back to the model
 * (which instructs Elle what to say). `cal` is injectable for tests.
 */
export async function handleBookIntroCall(
  args: Record<string, unknown>,
  ctx: BookingContext = {},
  cal: CalendarClient | null = defaultCalendar
): Promise<string> {
  const name = String(args.name ?? "").trim();
  const email = String(args.email ?? "").trim();
  const startRaw = String(args.start ?? "").trim();
  const topic = String(args.topic ?? "").trim();

  if (!name || !email || !startRaw) {
    return "Error: get the caller's name, email, and a specific proposed time before booking.";
  }
  if (!EMAIL_RE.test(email)) {
    return "Error: that email doesn't look valid. Confirm the caller's email address, then book.";
  }
  const start = parseStart(startRaw, config.booking.timezone);
  if (!start) {
    return "Error: couldn't read that start time. Ask the caller for a specific day and time and try again.";
  }
  const now = Date.now();
  if (start.getTime() < now - 60_000) {
    return (
      "Error: that time is in the past. Check the date with the caller (the current date and time " +
      "is in your context) and book a future slot."
    );
  }
  if (start.getTime() > now + MAX_LEAD_DAYS * 24 * 60 * 60_000) {
    return (
      `Error: that time is more than a year out, which usually means the year is wrong. ` +
      `Confirm the date with the caller and book again.`
    );
  }
  const end = new Date(start.getTime() + config.booking.durationMinutes * 60_000);

  let eventLink = "";
  let eventId = "";
  let placed = "recorded as a request (no calendar configured)";

  if (cal) {
    // Deliberately fails OPEN: if the availability check itself errors, the
    // booking still goes ahead. Losing a caller who is ready to commit is worse
    // than a hold the owner may have to move, and the owner confirms every one
    // of these by hand anyway. The failure is logged so it is not invisible.
    let free = true;
    try {
      free = await cal.isFree(start.toISOString(), end.toISOString());
    } catch (e) {
      console.error("Calendar freeBusy failed:", e);
    }
    if (!free) {
      return `That time is already busy on ${ownerName}'s calendar. Offer the caller a different time and book again.`;
    }
    try {
      const ev = await cal.createTentative({
        summary: `Intro call: ${name} + ${ownerName} (pending confirmation)`,
        description:
          `Tentative intro call requested via Elle.\n` +
          `Caller: ${name} <${email}>\n` +
          `Topic: ${topic || "n/a"}\n` +
          `Awaiting ${ownerName}'s confirmation.`,
        startISO: start.toISOString(),
        endISO: end.toISOString(),
        timezone: config.booking.timezone,
        attendeeEmail: email,
      });
      eventLink = ev.htmlLink;
      eventId = ev.id;
      placed = `penciled in as TENTATIVE on ${ownerName}'s calendar, pending his confirmation`;
    } catch (e) {
      console.error("Calendar event insert failed:", e);
      placed = "recorded as a request (calendar hold could not be created)";
    }
  }

  try {
    db.saveBooking({
      phoneNumber: ctx.fromNumber ?? null,
      callSid: ctx.callSid ?? null,
      name,
      email,
      topic: topic || null,
      startAt: start.toISOString(),
      status: "pending",
      eventLink: eventLink || null,
      eventId: eventId || null,
    });
  } catch (e) {
    // The slot lock fired: someone else took this time between our free/busy
    // check and this write. Undo the hold we just placed so the calendar does
    // not keep a duplicate, and send Elle back to offer another time.
    if (String(e).includes("UNIQUE") && String(e).includes("start_at")) {
      if (eventId && cal?.cancelEvent) {
        await cal.cancelEvent(eventId).catch((err) => console.error("rollback failed:", err));
      }
      return (
        `Someone else booked that exact time a moment ago. Apologize briefly, offer the caller ` +
        `another time, and book again.`
      );
    }
    throw e;
  }

  ctx.notify?.(`Intro call requested by ${name} for ${startRaw} (pending)`);
  // The owner hears about this immediately rather than at the end of the call:
  // the caller was just told he would confirm by email, and until now nothing
  // told him there was anything to confirm.
  notifyBooking({
    name,
    email,
    topic,
    startISO: start.toISOString(),
    phoneNumber: ctx.fromNumber ?? null,
    placed,
    eventLink: eventLink || null,
  });

  return (
    `Done: ${placed}. IMPORTANT — tell the caller this time is tentative until ${ownerName} confirms, ` +
    `and that he'll confirm by email to ${email}. Do not imply it is final. Then tell them they ` +
    `can text this same number any time to change or cancel it — say this HERE rather than in the ` +
    `greeting, because it only matters once they actually have a booking.`
  );
}
