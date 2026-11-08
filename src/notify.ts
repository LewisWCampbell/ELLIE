/**
 * Telling the owner what happened.
 *
 * The system was very good at capturing and completely silent about it: a
 * booking landed on the calendar as a tentative hold, Elle told the caller
 * "he'll confirm by email", and nothing reached the owner at all. The only way
 * to find out was to open an admin page and look. That is a promise the
 * product makes on every booking and did not keep.
 *
 * Everything here is fire-and-forget by construction. A notification that
 * fails, or an intake channel that is not configured, must never affect a live
 * call: the caller's experience does not depend on the owner's phone.
 */
import { config } from "./config";

/**
 * Send one line to the owner's private intake channel.
 *
 * Never throws and never rejects. Callers use `void notifyOwner(...)` and carry
 * on; the await is only ever for tests.
 */
export async function notifyOwner(text: string): Promise<boolean> {
  const chatId = config.telegram.ownerId;
  if (!chatId || !config.telegram.botToken) return false;
  try {
    // Imported at call time, not at module load. The intake channel module
    // reaches back into booking and SMS, so a static import here would close
    // the loop booking -> notify -> intake -> sms -> booking, and the whole
    // graph fails to initialize.
    const { tgSend } = await import("./telegram.js");
    await tgSend(chatId, text);
    return true;
  } catch (e) {
    console.error("[notify] owner notification failed:", e);
    return false;
  }
}

/** Local wall-clock, in the booking timezone, for a line the owner reads. */
export function ownerTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-US", {
      timeZone: config.booking.timezone,
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return iso;
  }
}

export interface BookingNotice {
  name: string;
  email: string;
  topic?: string | null;
  startISO: string;
  phoneNumber?: string | null;
  placed: string;
  eventLink?: string | null;
}

/**
 * A booking is the one event worth interrupting someone for, so it is sent the
 * moment it is written rather than batched into an end-of-call summary: the
 * call may run several more minutes, and the owner may want to act during it.
 */
export function notifyBooking(b: BookingNotice): void {
  const lines = [
    "New intro call booked",
    "",
    `Who    ${b.name} <${b.email}>`,
    `When   ${ownerTime(b.startISO)}`,
  ];
  if (b.topic) lines.push(`Topic  ${b.topic}`);
  if (b.phoneNumber) lines.push(`Phone  ${b.phoneNumber}`);
  lines.push(`Status ${b.placed}`);
  if (b.eventLink) lines.push("", b.eventLink);
  lines.push("", "They were told you would confirm by email.");
  void notifyOwner(lines.join("\n"));
}

export interface CallNotice {
  phoneNumber: string;
  seconds: number;
  turns: number;
  facts: Array<{ category: string; fact: string }>;
  booked: boolean;
}

/**
 * The end-of-call line. Short on purpose: enough to decide whether to look, no
 * more. A call with nothing learned and nothing said is not reported at all,
 * because a notification for every wrong number trains the owner to ignore the
 * channel, and then the ones that matter are ignored too.
 */
export function notifyCallEnded(c: CallNotice): void {
  if (c.booked) return; // the booking notice already went out; don't double up
  if (c.turns === 0 && c.facts.length === 0) return;

  const mins = Math.floor(c.seconds / 60);
  const secs = c.seconds % 60;
  const length = mins ? `${mins}m ${String(secs).padStart(2, "0")}s` : `${secs}s`;
  const lines = [`Call from ${c.phoneNumber} · ${length} · ${c.turns} turns`];
  if (c.facts.length) {
    lines.push("");
    for (const f of c.facts.slice(0, 8)) {
      lines.push(`· ${f.fact}`);
    }
  }
  void notifyOwner(lines.join("\n"));
}
