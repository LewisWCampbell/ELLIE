import { Router, Request, Response, NextFunction } from "express";
import twilio from "twilio";
import { config } from "./config";
import * as db from "./db";
import { provider } from "./llm/provider";
import { NeutralMessage, ToolDef, ToolResultMsg } from "./llm/types";
import { BOOK_INTRO_CALL, handleBookIntroCall, sharedCalendar } from "./booking/booking";
import { CHECK_AVAILABILITY, handleCheckAvailability } from "./booking/availability";
import { spanishEnabled } from "./persona";
import { logQuestion } from "./gaps";

/**
 * Texting with Elle. Callers (or anyone) can text the Twilio number to check,
 * reschedule, or cancel their intro-call booking, or make a new one; every
 * reply doubles as the confirmation text. The webhook answers with TwiML
 * (<Message>), so no outbound-SMS credentials are needed — Twilio delivers the
 * reply on the same inbound leg.
 *
 * Conversations are keyed by the sender's phone number and kept in memory with
 * a short TTL: long enough to feel like a thread, short enough to reset stale
 * context (and Cloud Run restarts clear it, which is fine — the DB holds the
 * bookings, which is the state that matters).
 */

const SMS_MAX_TOKENS = 300; // texts are short
const THREAD_TTL_MS = 6 * 60 * 60 * 1000; // 6h of quiet ends a thread
const THREAD_MAX_MESSAGES = 24;

interface SmsThread {
  messages: NeutralMessage[];
  lastAt: number;
}
const threads = new Map<string, SmsThread>();

function getThread(from: string): SmsThread {
  const existing = threads.get(from);
  if (existing && Date.now() - existing.lastAt < THREAD_TTL_MS) return existing;
  const fresh: SmsThread = { messages: [], lastAt: Date.now() };
  threads.set(from, fresh);
  return fresh;
}

// --- tools -------------------------------------------------------------------

const GET_BOOKINGS_TOOL: ToolDef = {
  name: "get_my_bookings",
  description:
    "Look up the texter's intro-call bookings (by their phone number). Call this before " +
    "rescheduling or canceling so you have the booking id and current time.",
  inputSchema: { type: "object", properties: {} },
};

const RESCHEDULE_TOOL: ToolDef = {
  name: "reschedule_booking",
  description:
    "Move an existing intro-call booking to a new start time. Only after the texter has " +
    "confirmed the new time. The moved slot goes back to pending until the owner re-confirms.",
  inputSchema: {
    type: "object",
    properties: {
      booking_id: { type: "number", description: "The booking id from get_my_bookings." },
      new_start: {
        type: "string",
        description: "New start, ISO 8601 with offset, e.g. 2026-08-25T14:00:00-07:00.",
      },
    },
    required: ["booking_id", "new_start"],
  },
};

const CANCEL_TOOL: ToolDef = {
  name: "cancel_booking",
  description:
    "Cancel an existing intro-call booking. Only after the texter has clearly asked to cancel " +
    "and you have confirmed which booking they mean.",
  inputSchema: {
    type: "object",
    properties: {
      booking_id: { type: "number", description: "The booking id from get_my_bookings." },
    },
    required: ["booking_id"],
  },
};

const SMS_TOOLS: ToolDef[] = [
  GET_BOOKINGS_TOOL,
  RESCHEDULE_TOOL,
  CANCEL_TOOL,
  { name: CHECK_AVAILABILITY.name, description: CHECK_AVAILABILITY.description, inputSchema: CHECK_AVAILABILITY.parameters },
  { name: BOOK_INTRO_CALL.name, description: BOOK_INTRO_CALL.description, inputSchema: BOOK_INTRO_CALL.parameters },
];

function fmtWhen(iso: string): string {
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

async function runSmsTool(
  name: string,
  args: Record<string, unknown>,
  from: string
): Promise<string> {
  const cal = sharedCalendar();
  switch (name) {
    case "get_my_bookings": {
      const rows = db.getBookingsForPhone(from);
      if (!rows.length) return "No bookings found for this phone number.";
      return rows
        .map(
          (b) =>
            `#${b.id}: ${b.name} — ${fmtWhen(b.start_at)} (${b.status})` +
            (b.topic ? ` — topic: ${b.topic}` : "")
        )
        .join("\n");
    }

    case "reschedule_booking": {
      const id = Number(args.booking_id);
      const b = Number.isInteger(id) ? db.getBookingById(id) : undefined;
      if (!b) return "Error: no booking with that id.";
      if (b.phone_number && b.phone_number !== from) {
        return "Error: that booking belongs to a different phone number.";
      }
      if (b.status === "canceled") return "Error: that booking was already canceled.";
      const start = new Date(String(args.new_start ?? ""));
      if (Number.isNaN(start.getTime())) {
        return "Error: couldn't read the new time. Ask for a specific day and time.";
      }
      const end = new Date(start.getTime() + config.booking.durationMinutes * 60_000);
      if (cal) {
        try {
          if (!(await cal.isFree(start.toISOString(), end.toISOString()))) {
            return `That time is busy on ${config.ownerName}'s calendar. Offer another time.`;
          }
          if (b.event_id && cal.patchEventStart) {
            await cal.patchEventStart(
              b.event_id,
              start.toISOString(),
              end.toISOString(),
              config.booking.timezone
            );
          }
        } catch (e) {
          console.error("SMS reschedule calendar update failed:", e);
        }
      }
      db.updateBookingTime(id, start.toISOString());
      return (
        `Done: booking #${id} moved to ${fmtWhen(start.toISOString())}. It is TENTATIVE again ` +
        `until ${config.ownerName} re-confirms — tell them he'll confirm by email.`
      );
    }

    case "cancel_booking": {
      const id = Number(args.booking_id);
      const b = Number.isInteger(id) ? db.getBookingById(id) : undefined;
      if (!b) return "Error: no booking with that id.";
      if (b.phone_number && b.phone_number !== from) {
        return "Error: that booking belongs to a different phone number.";
      }
      if (b.status === "canceled") return "That booking is already canceled.";
      if (b.event_id && cal?.cancelEvent) {
        try {
          await cal.cancelEvent(b.event_id);
        } catch (e) {
          console.error("SMS cancel calendar delete failed:", e);
        }
      }
      db.updateBookingStatus(id, "canceled");
      return `Done: booking #${id} (${fmtWhen(b.start_at)}) is canceled. Confirm it to them.`;
    }

    case "check_availability":
      return handleCheckAvailability(args);

    case "book_intro_call":
      return handleBookIntroCall(args, { fromNumber: from });

    default:
      return `Unknown tool: ${name}`;
  }
}

// --- persona -----------------------------------------------------------------

function smsPersona(): string {
  const owner = config.ownerName;
  const elle = config.assistantName;
  return (
    `You are ${elle}, ${owner}'s AI assistant, texting over SMS from his number. People text ` +
    `to manage their intro-call booking with ${owner} (check it, reschedule it, cancel it) or ` +
    `to set one up. Be warm and efficient.\n\n` +
    `Rules:\n` +
    `- SMS style: 1-3 short sentences, plain text only, no markdown or emoji. One question at a time.\n` +
    `- Before changing anything, call get_my_bookings to see their bookings. If they have ` +
    `several, ask which one they mean.\n` +
    `- Read the new time back and get a clear yes before reschedule_booking or cancel_booking.\n` +
    `- Every change is TENTATIVE until ${owner} confirms by email — always say so.\n` +
    `- Your reply IS their confirmation text: after a change, restate exactly what changed and ` +
    `the day/time in ${config.booking.timezone}.\n` +
    `- If they ask about ${owner} himself — his background, skills, or work — give a one-line ` +
    `answer and include this exact link to explore more: https://lewiswcampbell.com ` +
    `(never any other URL). Calling this same number to talk it through is a good suggestion too.\n` +
    `- Never invent bookings or times; only state what the tools returned.` +
    (spanishEnabled()
      ? `\n- If they text in Spanish, reply entirely in natural Spanish (same rules); if they ` +
        `switch languages, switch with them.`
      : ``)
  );
}

function smsContext(from: string): string {
  let ctx =
    `Right now it is ${new Date().toISOString()} (UTC). ${config.ownerName}'s timezone is ` +
    `${config.booking.timezone}; intro calls are ${config.booking.durationMinutes} minutes. ` +
    `Convert times the texter gives you to ISO 8601 with the correct offset for that timezone.`;
  const o = db.getOutreach(from);
  if (o) {
    ctx +=
      `\n\nOutreach context: at ${config.ownerName}'s request you already texted this person ` +
      `first. Their name is ${o.name}` +
      (o.topic ? `; the topic is ${o.topic}` : "") +
      (o.message ? `. Your first message to them was: "${o.message}"` : ".") +
      ` They are replying to that message, so greet them by name, don't re-introduce the topic ` +
      `from scratch, and steer toward finding a time for the intro call with ${config.ownerName}.`;
  }
  return ctx;
}

// --- webhook -----------------------------------------------------------------

const MessagingResponse = twilio.twiml.MessagingResponse;
export const smsRouter = Router();

/* Reachable without a token only under ALLOW_UNSIGNED_TWILIO=1; boot exits otherwise (src/index.ts). */
function validateTwilio(req: Request, res: Response, next: NextFunction): void {
  if (!config.twilioAuthToken) return next();
  const signature = req.header("X-Twilio-Signature") ?? "";
  const url = `${config.publicUrl}${req.originalUrl}`;
  if (!twilio.validateRequest(config.twilioAuthToken, signature, url, req.body ?? {})) {
    res.status(403).send("Invalid Twilio signature");
    return;
  }
  next();
}

// CTIA-standard keywords. STOP must halt everything; START re-subscribes; HELP
// must return identification + support + rates + opt-out language.
const STOP_WORDS = new Set(["STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"]);
const START_WORDS = new Set(["START", "UNSTOP"]);
const HELP_WORDS = new Set(["HELP", "INFO"]);

const TERMS_URL = "https://lewiswcampbell.com/sms-terms.html";
const PRIVACY_URL = "https://lewiswcampbell.com/privacy.html";

const DISCLOSURE =
  "Msg frequency varies. Msg & data rates may apply. Reply STOP to opt out, HELP for help. " +
  `Terms: ${TERMS_URL} Privacy: ${PRIVACY_URL}`;

/**
 * Who is responsible for this messaging program, in the two messages that have
 * to say so. Falls back to the assistant alone when no brand is configured, so
 * a deployment without BRAND_NAME reads exactly as it did before.
 */
function operatedBy(): string {
  return config.brandName ? ` Messages are sent by ${config.brandName}.` : "";
}

/**
 * Registered keyword replies are capped at 320 characters by the campaign
 * registry, so HELP has a budget rather than room for a sentence about what
 * Elle can do. What has to survive, in this order, is identification (who is
 * responsible), how to reach a human, frequency, rates, the opt-out
 * instruction, and the terms link. Everything else is decoration.
 *
 * Exported for the length assertion in sms.verify.ts: a HELP reply that
 * overflows is not a runtime error, it is a campaign that gets rejected weeks
 * later, so the check belongs in the suite that runs before every deploy.
 */
export const KEYWORD_REPLY_MAX = 320;

export function helpText(): string {
  const brand = config.brandName ? ` for ${config.brandName}` : "";
  return (
    `${config.assistantName} is the automated assistant${brand}, for appointments with ` +
    `${config.ownerName}. Text or call this number for help. Msg frequency varies. ` +
    `Msg & data rates may apply. Reply STOP to opt out. Terms: ${TERMS_URL} Privacy: ${PRIVACY_URL}`
  );
}

/**
 * Send an outbound SMS via Twilio's REST API. Used only for the consent
 * confirmation text after a caller verbally opts in on a phone call (a carrier
 * requirement for verbal opt-in). Needs TWILIO_ACCOUNT_SID and
 * TWILIO_FROM_NUMBER alongside the auth token; silently no-ops when outbound
 * isn't configured.
 */
export async function sendSms(to: string, body: string): Promise<boolean> {
  const sid = config.twilioAccountSid;
  const from = config.twilioFromNumber;
  if (!sid || !from || !config.twilioAuthToken) return false;
  const res = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${sid}/Messages.json`, {
    method: "POST",
    headers: {
      Authorization: "Basic " + Buffer.from(`${sid}:${config.twilioAuthToken}`).toString("base64"),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({ To: to, From: from, Body: body }).toString(),
  });
  if (!res.ok) throw new Error(`Twilio send ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return true;
}

/**
 * The enrollment-confirmation text sent after a verbal opt-in on a call. This
 * is the message that documents consent, so it names who is responsible and
 * repeats every disclosure the caller heard read aloud.
 *
 * Separate from the sender so its length can be asserted offline: it is
 * registered as a sample and shares HELP's 320 character cap.
 */
export function consentConfirmationText(): string {
  return (
    `You're enrolled in booking-related texts from ${config.assistantName}, ${config.ownerName}'s AI assistant.` +
    `${operatedBy()} Msg frequency varies. Msg & data rates may apply. Reply HELP for help, STOP to opt out. ` +
    `Terms: ${TERMS_URL} Privacy: ${PRIVACY_URL}`
  );
}

export async function sendConsentConfirmation(to: string): Promise<void> {
  const sent = await sendSms(to, consentConfirmationText());
  if (sent) console.log(`[sms] consent confirmation sent to ${to}`);
}

smsRouter.post("/sms", validateTwilio, async (req: Request, res: Response) => {
  const from = String(req.body.From ?? "unknown");
  const body = String(req.body.Body ?? "").trim();
  const twiml = new MessagingResponse();

  if (!body) {
    res.type("text/xml").send(twiml.toString());
    return;
  }

  // --- A2P compliance gate, before any AI is involved -----------------------
  const keyword = body.toUpperCase().replace(/[.!]+$/, "").trim();

  if (STOP_WORDS.has(keyword)) {
    // Record the opt-out and send nothing: Twilio's default opt-out handling
    // delivers the mandated confirmation and blocks further sends to this number.
    db.setSmsConsent(from, "opted_out", "sms_stop");
    threads.delete(from);
    console.log(`[sms] ${from} opted out (${keyword})`);
    res.type("text/xml").send(twiml.toString());
    return;
  }

  if (HELP_WORDS.has(keyword)) {
    twiml.message(helpText());
    res.type("text/xml").send(twiml.toString());
    return;
  }

  const consent = db.getSmsConsent(from);
  if (consent?.status === "opted_out") {
    if (START_WORDS.has(keyword) || keyword === "YES") {
      db.setSmsConsent(from, "opted_in", "sms_start");
      console.log(`[sms] ${from} opted back in (${keyword})`);
      twiml.message(
        `You're opted back in to texts from ${config.assistantName}, ${config.ownerName}'s AI assistant. ${DISCLOSURE}`
      );
    }
    // Anything else from an opted-out number: stay silent (carriers require it;
    // Twilio blocks outbound to them regardless).
    res.type("text/xml").send(twiml.toString());
    return;
  }
  // Texting us first is the opt-in. Record it once (or refresh a prior state).
  if (!consent || consent.status !== "opted_in") {
    db.setSmsConsent(from, "opted_in", "inbound_text");
  }

  // If this number was on the owner's outreach list, their first text back
  // closes the loop — mark it so check_outreach on Telegram shows "replied".
  const outreach = db.getOutreach(from);
  if (outreach && outreach.status === "sent") {
    db.updateOutreachStatus(from, "replied");
    console.log(`[sms] outreach reply from ${outreach.name} (${from})`);
  }

  const thread = getThread(from);
  const isNewThread = thread.messages.length === 0;
  thread.messages.push({ role: "user", text: body });
  thread.lastAt = Date.now();
  db.saveTranscript(`sms:${from}`, "caller", body);
  logQuestion("sms", body);
  console.log(`[sms] ${from}: "${body}"`);

  let replyText =
    "Sorry, something went wrong on my end. Please try again, or call this number to talk to me.";
  try {
    // Tool loop within this webhook; history carries only user/assistant text.
    const working: NeutralMessage[] = [...thread.messages];
    let resp = await provider.chat({
      systemStable: smsPersona(),
      systemDynamic: smsContext(from),
      messages: working,
      tools: SMS_TOOLS,
      maxTokens: SMS_MAX_TOKENS,
    });
    for (let hop = 0; hop < 5 && resp.toolCalls.length > 0; hop++) {
      working.push({ role: "assistant", text: resp.text, toolCalls: resp.toolCalls, raw: resp.raw });
      const results: ToolResultMsg[] = [];
      for (const call of resp.toolCalls) {
        let output: string;
        try {
          output = await runSmsTool(call.name, call.input, from);
        } catch (e) {
          console.error(`[sms] tool ${call.name} failed:`, e);
          output = "Error: that action failed. Apologize and suggest calling instead.";
        }
        results.push({ id: call.id, content: output, isError: output.startsWith("Error") });
      }
      working.push({ role: "tool_results", results });
      resp = await provider.chat({
        systemStable: smsPersona(),
        systemDynamic: smsContext(from),
        messages: working,
        tools: SMS_TOOLS,
        maxTokens: SMS_MAX_TOKENS,
      });
    }
    if (resp.text.trim()) replyText = resp.text.trim();
  } catch (e) {
    console.error("[sms] turn failed:", e);
  }

  thread.messages.push({ role: "assistant", text: replyText, toolCalls: [] });
  if (thread.messages.length > THREAD_MAX_MESSAGES) {
    thread.messages = thread.messages.slice(-THREAD_MAX_MESSAGES);
  }
  // First reply of a fresh thread carries the required disclosure.
  const outbound = isNewThread ? `${replyText}\n\n${DISCLOSURE}` : replyText;
  db.saveTranscript(`sms:${from}`, "assistant", outbound);
  console.log(`[sms] elle -> ${from}: "${outbound}"`);

  twiml.message(outbound);
  res.type("text/xml").send(twiml.toString());
});
