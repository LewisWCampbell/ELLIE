/**
 * Appointment reminders.
 *
 * The A2P campaign registers four message types and this is one of them: "a
 * reminder about that appointment". It was registered and never built, which
 * is the harmless direction of that mismatch, but it is also the cheapest
 * no-show reduction available and the carriers have already approved it.
 *
 * Two rules shape the implementation. Consent is checked at send time, not at
 * booking time, because someone can text STOP in between and a reminder is
 * exactly the message they were opting out of. And the claim is taken BEFORE
 * the send: with more than one instance running, two containers can both find
 * the same due booking, and a duplicate reminder is worse than a missed one.
 */
import { config } from "../config";
import * as db from "../db";
import { sendSms } from "../sms";
import { notifyOwner, ownerTime } from "../notify";

/** How far ahead of the appointment the reminder goes out. */
const LEAD_HOURS = 24;

/**
 * Reminders are only sent inside this window before the appointment. The lower
 * bound is what stops a job that has not run for a day from "reminding" people
 * about appointments that already started.
 */
const WINDOW_HOURS = 26;

export interface ReminderRun {
  considered: number;
  sent: number;
  skippedNoConsent: number;
  skippedNoNumber: number;
  claimedElsewhere: number;
  failed: number;
}

export function reminderText(name: string, startISO: string): string {
  const first = name.trim().split(/\s+/)[0] || name;
  return (
    `Hi ${first}, a reminder from ${config.assistantName}: your intro call with ` +
    `${config.ownerName} is ${ownerTime(startISO)}. Reply here to change or cancel it, ` +
    `or STOP to opt out.`
  );
}

/**
 * Send every reminder that is due. Safe to run on any schedule and safe to run
 * concurrently: the claim in the database decides who sends.
 */
export async function sendDueReminders(now = new Date()): Promise<ReminderRun> {
  const run: ReminderRun = {
    considered: 0,
    sent: 0,
    skippedNoConsent: 0,
    skippedNoNumber: 0,
    claimedElsewhere: 0,
    failed: 0,
  };

  const from = new Date(now.getTime() + (LEAD_HOURS - (WINDOW_HOURS - LEAD_HOURS)) * 3_600_000);
  const until = new Date(now.getTime() + WINDOW_HOURS * 3_600_000);
  const due = db.getBookingsDueForReminder(now.toISOString(), until.toISOString());
  void from;

  for (const b of due) {
    run.considered++;

    if (!b.phone_number) {
      // Booked from the web chat, so there is no number to text. Claimed
      // anyway, or every future run reconsiders it forever.
      db.claimReminder(b.id);
      run.skippedNoNumber++;
      continue;
    }

    const consent = db.getSmsConsent(b.phone_number);
    if (consent?.status === "opted_out") {
      db.claimReminder(b.id);
      run.skippedNoConsent++;
      continue;
    }

    // Claim first. If another instance got there, it is sending this one.
    if (!db.claimReminder(b.id)) {
      run.claimedElsewhere++;
      continue;
    }

    try {
      const ok = await sendSms(b.phone_number, reminderText(b.name, b.start_at));
      if (ok) run.sent++;
      else run.failed++;
    } catch (e) {
      console.error(`[reminders] send failed for booking ${b.id}:`, e);
      run.failed++;
    }
  }

  if (run.sent > 0) {
    void notifyOwner(
      `Sent ${run.sent} appointment reminder${run.sent === 1 ? "" : "s"} for tomorrow.`
    );
  }
  return run;
}
