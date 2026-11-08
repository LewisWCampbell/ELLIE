import type { CalendarClient } from "./calendar";

/**
 * The booking calendar's end-to-end self-check: read, create, confirm busy,
 * move, delete.
 *
 * This lives here rather than in the command-line tool because the tool runs as
 * whoever is at a terminal, and that is the wrong identity. Elle books as the
 * service account, so the only check that answers the real question is one that
 * runs inside the service. The same function backs both: /admin/calendar-check
 * (as the service account, the authoritative answer) and the local tool (as
 * you, useful for confirming the calendar id and that the API is on).
 *
 * It cleans up after itself. The hold it creates is far enough out that it
 * cannot collide with a real meeting, and it is deleted before returning.
 */

export interface CheckStep {
  name: string;
  status: "ok" | "warn" | "fail";
  detail: string;
}

export interface CheckResult {
  ok: boolean;
  steps: CheckStep[];
  /** Set when a step failed in a way worth explaining at length. */
  diagnosis?: string;
}

/** Far enough ahead that the probe cannot land on a real meeting. */
const PROBE_DAYS_OUT = 180;

/**
 * A reserved example domain, so the address can never receive anything. A guest
 * is attached because every real booking attaches one, and that is exactly the
 * operation Google refuses for a service account on a shared calendar. A probe
 * without a guest takes a different path from the thing being checked.
 */
const PROBE_GUEST = "elle-smoke-test@example.com";

/**
 * Tell the two 403s apart. They look alike and have opposite fixes: one is
 * about the token, the other about the calendar.
 */
function diagnose(message: string): string | undefined {
  if (/insufficient authentication scopes|insufficient permission/i.test(message)) {
    return (
      "SCOPE, not sharing. The credentials in use were issued without the " +
      "Calendar scope, so the request never reached the question of access. On " +
      "Cloud Run this means the service account needs the calendar scope; note " +
      "that Cloud Run service accounts get cloud-platform, which covers Calendar " +
      "only if the Calendar API is enabled on the project. Enable it with: " +
      "gcloud services enable calendar-json.googleapis.com"
    );
  }
  if (/forbidden|permission|403/i.test(message)) {
    return (
      "ACCESS. The account can authenticate but is not allowed on this calendar. " +
      "Share the calendar with it at 'Make changes to events'. Read-only is the " +
      "usual mistake and fails exactly here."
    );
  }
  if (/not shared|cannot see this calendar|could not read/i.test(message)) {
    return (
      "NOT SHARED. The account authenticated fine but this calendar is not visible " +
      "to it. Open the calendar's 'Settings and sharing', and under 'Share with " +
      "specific people' add the service account address with permission 'Make " +
      "changes to events'. Sharing can take a couple of minutes to take effect."
    );
  }
  if (/not found|404/i.test(message)) {
    return (
      "NOT FOUND for this account. Either the calendar is not shared with it, or " +
      "BOOKING_CALENDAR_ID names a calendar that does not exist. Sharing is the " +
      "usual cause when the id is an address you own: owning a calendar does not " +
      "give the service account access to it. Add the service account under " +
      "'Share with specific people' at 'Make changes to events'."
    );
  }
  return undefined;
}

export async function runCalendarCheck(
  cal: CalendarClient,
  timezone: string,
  durationMinutes: number,
  now: Date = new Date()
): Promise<CheckResult> {
  const steps: CheckStep[] = [];
  const finish = (diagnosis?: string): CheckResult => ({
    ok: steps.every((s) => s.status !== "fail"),
    steps,
    diagnosis,
  });

  const day = new Date(now.getTime() + PROBE_DAYS_OUT * 86_400_000);
  day.setUTCHours(11, 30, 0, 0);
  const start = day;
  const end = new Date(start.getTime() + durationMinutes * 60_000);

  // 1. Read. Proves auth, the API being enabled, and the id resolving.
  let free: boolean;
  try {
    free = await cal.isFree(start.toISOString(), end.toISOString());
    steps.push({ name: "read free/busy", status: "ok", detail: free ? "slot is free" : "slot is busy" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    steps.push({ name: "read free/busy", status: "fail", detail: msg });
    return finish(diagnose(msg));
  }

  // 2. Write, with a guest, exactly as a booking does.
  let created: { htmlLink: string; id: string; guestAttached?: boolean };
  try {
    created = await cal.createTentative({
      summary: "Elle booking check (safe to delete)",
      description: "Created by the booking self-check to verify write access. Deleted automatically.",
      startISO: start.toISOString(),
      endISO: end.toISOString(),
      timezone,
      attendeeEmail: PROBE_GUEST,
    });
    steps.push({ name: "create tentative hold", status: "ok", detail: `event ${created.id}` });
    steps.push({
      name: "attach the caller as a guest",
      status: created.guestAttached === false ? "warn" : "ok",
      detail:
        created.guestAttached === false
          ? "refused, so the hold was created without one. Expected on a personal calendar: " +
            "guests need Domain-Wide Delegation, which needs a Workspace domain. Holds still " +
            "work and the caller's email is in the event description."
          : "attached",
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    steps.push({ name: "create tentative hold", status: "fail", detail: msg });
    return finish(diagnose(msg));
  }

  // 3. The hold must read as busy, or the conflict check protects nothing.
  try {
    const busy = !(await cal.isFree(start.toISOString(), end.toISOString()));
    steps.push({
      name: "hold reads as busy",
      status: busy ? "ok" : "warn",
      detail: busy
        ? "double-booking protection works"
        : "the hold does NOT read as busy, so a second booking in this slot would not be refused",
    });
  } catch (e) {
    steps.push({
      name: "hold reads as busy",
      status: "warn",
      detail: e instanceof Error ? e.message : String(e),
    });
  }

  // 4. Move it, which is what a reschedule by text does.
  if (cal.patchEventStart) {
    const moved = new Date(start.getTime() + 60 * 60_000);
    try {
      await cal.patchEventStart(
        created.id,
        moved.toISOString(),
        new Date(moved.getTime() + durationMinutes * 60_000).toISOString(),
        timezone
      );
      steps.push({ name: "move the hold", status: "ok", detail: "reschedule path works" });
    } catch (e) {
      steps.push({
        name: "move the hold",
        status: "fail",
        detail: e instanceof Error ? e.message : String(e),
      });
    }
  }

  // 5. Clean up. A check that litters the owner's calendar is worse than none.
  if (cal.cancelEvent) {
    try {
      await cal.cancelEvent(created.id);
      steps.push({ name: "delete the hold", status: "ok", detail: "nothing left behind" });
    } catch (e) {
      steps.push({
        name: "delete the hold",
        status: "fail",
        detail: `could not delete ${created.id}, remove it by hand: ${created.htmlLink} (${
          e instanceof Error ? e.message : String(e)
        })`,
      });
    }
  }

  return finish();
}
