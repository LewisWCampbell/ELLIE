import { GoogleAuth } from "google-auth-library";
import { config } from "../config";

/**
 * Minimal Google Calendar client for tentative intro-call holds. Auth is ADC:
 * the Cloud Run service account must have the calendar shared with it ("Make
 * changes to events"), and the Calendar API enabled. Called over REST with a
 * bearer token so we don't pull the full googleapis SDK.
 *
 * Behind an interface so the booking handler can be tested offline with a fake.
 */
export interface TentativeDetails {
  summary: string;
  description: string;
  startISO: string;
  endISO: string;
  timezone: string;
  /** Caller's email, attached as a guest but NOT notified until the owner sends. */
  attendeeEmail?: string;
}

export interface BusyRange {
  start: string;
  end: string;
}

export interface CalendarClient {
  /** True if nothing is busy on the calendar between the two times. */
  isFree(startISO: string, endISO: string): Promise<boolean>;
  /**
   * Every busy block between the two times, across all configured calendars.
   * Optional so fakes can omit it; availability falls back to isFree without it.
   */
  busy?(startISO: string, endISO: string): Promise<BusyRange[]>;
  /**
   * Create a TENTATIVE event; returns its link + id. `guestAttached` is false
   * when the caller could not be added as a guest and the hold was created
   * without one, which is the normal outcome on a personal calendar.
   */
  createTentative(
    d: TentativeDetails
  ): Promise<{ htmlLink: string; id: string; guestAttached?: boolean }>;
  /** Move an event to a new time, back to tentative (optional: fakes may omit). */
  patchEventStart?(eventId: string, startISO: string, endISO: string, timezone: string): Promise<void>;
  /** Cancel (delete) an event (optional: fakes may omit). */
  cancelEvent?(eventId: string): Promise<void>;
}

/**
 * Google refuses to let a plain service account add guests to an event:
 *
 *   403 Service accounts cannot invite attendees without Domain-Wide
 *       Delegation of Authority.
 *
 * That restriction applies to any calendar simply SHARED with the service
 * account, which is the normal setup for a personal Gmail calendar. Delegation
 * requires a Workspace domain, so for a personal calendar it is not an option.
 */
const NO_ATTENDEES_RE = /domain-wide delegation|cannot invite attendees/i;

/** Real ADC-authed client, or null when no calendar is configured. */
export function createCalendarClient(
  getToken?: () => Promise<string>
): CalendarClient | null {
  if (!config.booking.calendarId) return null;

  const calendarId = config.booking.calendarId;
  const encId = encodeURIComponent(calendarId);
  let auth: GoogleAuth | null = null;

  const bearer =
    getToken ??
    (async function bearer(): Promise<string> {
      auth ??= new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/calendar"] });
      const client = await auth.getClient();
      const token = await client.getAccessToken();
      if (!token.token) throw new Error("no Google access token for Calendar");
      return token.token;
    });

  // The booking calendar plus any read-only calendars configured for
  // conflicts. One request covers all of them; freeBusy answers per
  // calendar, and a slot is free only if every one of them is free.
  async function busyRanges(startISO: string, endISO: string): Promise<BusyRange[]> {
    const ids = [calendarId, ...config.booking.busyCalendars.filter((id) => id !== calendarId)];
    const res = await fetch("https://www.googleapis.com/calendar/v3/freeBusy", {
      method: "POST",
      headers: { Authorization: `Bearer ${await bearer()}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        timeMin: startISO,
        timeMax: endISO,
        items: ids.map((id) => ({ id })),
      }),
    });
    if (!res.ok) throw new Error(`freeBusy ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const data = (await res.json()) as {
      calendars?: Record<string, { busy?: BusyRange[]; errors?: Array<{ reason?: string }> }>;
    };

    for (const id of ids) {
      const e = data.calendars?.[id];
      // freeBusy answers 200 even when it could not read a calendar, and
      // reports that per-calendar instead: an `errors` array, or no entry at
      // all. Both fall through to `busy ?? []`, which is empty, which reads
      // as FREE — so an unreadable calendar once reported as wide open and
      // the conflict check passed on a calendar nobody could see. An
      // unreadable calendar is an error, not a free one. That applies to
      // every id here: a personal calendar that stopped being shared must
      // not silently drop out of the check that exists to respect it.
      if (!e) {
        throw new Error(`freeBusy returned no data for ${id}. The account cannot see this calendar.`);
      }
      if (e.errors?.length) {
        const reasons = e.errors.map((x) => x.reason ?? "unknown").join(", ");
        throw new Error(
          `freeBusy could not read ${id} (${reasons}). ` +
            `Usually this means the calendar is not shared with this account.`
        );
      }
    }
    return ids.flatMap((id) => data.calendars?.[id]?.busy ?? []);
  }

  return {
    async isFree(startISO, endISO): Promise<boolean> {
      return (await busyRanges(startISO, endISO)).length === 0;
    },

    busy: busyRanges,

    async createTentative(d): Promise<{ htmlLink: string; id: string; guestAttached?: boolean }> {
      const insert = async (withAttendee: boolean): Promise<Response> =>
        fetch(`https://www.googleapis.com/calendar/v3/calendars/${encId}/events?sendUpdates=none`, {
          method: "POST",
          headers: { Authorization: `Bearer ${await bearer()}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            summary: d.summary,
            description: d.description,
            status: "tentative",
            start: { dateTime: d.startISO, timeZone: d.timezone },
            end: { dateTime: d.endISO, timeZone: d.timezone },
            // Attach the caller as a guest. sendUpdates=none means they are NOT
            // emailed now; the owner confirming and hitting Send invites them.
            ...(withAttendee && d.attendeeEmail ? { attendees: [{ email: d.attendeeEmail }] } : {}),
          }),
        });

      let res = await insert(true);
      let guestAttached = Boolean(d.attendeeEmail);

      // Retry without the guest rather than lose the hold. On a calendar merely
      // shared with the service account, which is the normal setup for a
      // personal Gmail calendar, attaching a guest is refused outright. Failing
      // the whole insert there would mean every booking degrades to a lead and
      // nothing ever reaches the calendar, while the caller has been told their
      // time is held. The hold is what matters; the caller's email is in the
      // event description either way, and the owner adds them as a guest when
      // they confirm and send the invite.
      if (!res.ok && d.attendeeEmail) {
        const body = await res.text();
        if (res.status === 403 && NO_ATTENDEES_RE.test(body)) {
          console.warn(
            "[calendar] this account cannot add guests to events (needs Domain-Wide " +
              "Delegation, which a personal calendar cannot have). Creating the hold " +
              "without the guest; the caller's email is in the event description."
          );
          guestAttached = false;
          res = await insert(false);
        } else {
          throw new Error(`events.insert ${res.status}: ${body.slice(0, 200)}`);
        }
      }

      if (!res.ok) throw new Error(`events.insert ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const ev = (await res.json()) as { htmlLink?: string; id?: string };
      return { htmlLink: ev.htmlLink ?? "", id: ev.id ?? "", guestAttached };
    },

    async patchEventStart(eventId, startISO, endISO, timezone): Promise<void> {
      const res = await fetch(
        `https://www.googleapis.com/calendar/v3/calendars/${encId}/events/${encodeURIComponent(eventId)}?sendUpdates=none`,
        {
          method: "PATCH",
          headers: { Authorization: `Bearer ${await bearer()}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            status: "tentative", // a moved hold goes back to awaiting confirmation
            start: { dateTime: startISO, timeZone: timezone },
            end: { dateTime: endISO, timeZone: timezone },
          }),
        }
      );
      if (!res.ok) throw new Error(`events.patch ${res.status}: ${(await res.text()).slice(0, 200)}`);
    },

    async cancelEvent(eventId): Promise<void> {
      const res = await fetch(
        `https://www.googleapis.com/calendar/v3/calendars/${encId}/events/${encodeURIComponent(eventId)}?sendUpdates=none`,
        { method: "DELETE", headers: { Authorization: `Bearer ${await bearer()}` } }
      );
      // 410 = already gone; that's fine for a cancel.
      if (!res.ok && res.status !== 410) {
        throw new Error(`events.delete ${res.status}: ${(await res.text()).slice(0, 200)}`);
      }
    },
  };
}
