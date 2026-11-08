import { Router, Request, Response } from "express";
import crypto from "crypto";
import { config } from "./config";
import * as db from "./db";
import { provider } from "./llm/provider";
import { NeutralMessage, ToolDef, ToolResultMsg } from "./llm/types";
import { sendSms } from "./sms";
import {
  currentVoiceProvider,
  setVoiceProvider,
  voiceProviderMinutesLeft,
  voiceProviderOverridden,
} from "./realtime/factory";

/**
 * The owner's private intake channel. Lewis messages Elle on Telegram with a
 * name, a phone number, and what the conversation is about; she confirms the
 * details, sends that person a compliance-complete first-contact SMS from the
 * Twilio number, and records the outreach so the SMS agent recognizes the
 * reply. He can also ask her how any outreach is going.
 *
 * Security model: the webhook path is unguessable only by convention, so two
 * real gates protect it — Telegram's secret_token header (set at setWebhook
 * time, echoed on every update) and TELEGRAM_OWNER_ID (updates from any other
 * account are dropped). Until the owner id is configured, the bot answers any
 * sender with their numeric id so setup is self-serve, and does nothing else.
 */

const TG_MAX_TOKENS = 400;
const THREAD_TTL_MS = 6 * 60 * 60 * 1000;
const THREAD_MAX_MESSAGES = 30;

interface TgThread {
  messages: NeutralMessage[];
  lastAt: number;
}
let ownerThread: TgThread = { messages: [], lastAt: 0 };

function getOwnerThread(): TgThread {
  if (Date.now() - ownerThread.lastAt >= THREAD_TTL_MS) {
    ownerThread = { messages: [], lastAt: Date.now() };
  }
  return ownerThread;
}

// --- Telegram API ------------------------------------------------------------

export async function tgSend(chatId: number | string, text: string): Promise<void> {
  const token = config.telegram.botToken;
  if (!token) return;
  const res = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text }),
  });
  if (!res.ok) {
    console.error(`[telegram] sendMessage ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
}

// --- tools -------------------------------------------------------------------

const SEND_OUTREACH_TOOL: ToolDef = {
  name: "send_outreach",
  description:
    "Send the first-contact SMS to someone Lewis wants Elle to reach out to. Only after Lewis " +
    "has confirmed the name, phone number, and topic in this conversation. Composes and sends " +
    "the compliant intro text and records the outreach.",
  inputSchema: {
    type: "object",
    properties: {
      name: { type: "string", description: "The person's name, as Lewis gave it." },
      phone: {
        type: "string",
        description: "Their mobile number. E.164 preferred (+15551234567); US 10-digit is accepted.",
      },
      topic: {
        type: "string",
        description: "What the intro call is about, in a few words (e.g. 'the analytics role at Acme').",
      },
    },
    required: ["name", "phone"],
  },
};

const CHECK_OUTREACH_TOOL: ToolDef = {
  name: "check_outreach",
  description:
    "Status report on Elle's outreach: who she has texted at Lewis's request, whether they " +
    "replied, and any bookings on their number. Call this when Lewis asks how outreach is going.",
  inputSchema: { type: "object", properties: {} },
};

const VOICE_MODE_TOOL: ToolDef = {
  name: "voice_mode",
  description:
    "Check or switch which voice architecture answers the phone. Without 'mode', reports the " +
    "current one. 'cascade' is the production pipeline (cheap, tool-reliable). 'gcp' is the " +
    "fused duplex model on Vertex AI: natural turn-taking, no extra vendor, inside the GCP " +
    "BAA — the preferred way to show off. 'demo' is the same class of model on OpenAI, kept " +
    "as a fallback. Switching applies from the NEXT call; calls in progress are unaffected. " +
    "Both fused modes revert on their own after 'minutes' so neither can be left on by " +
    "accident.",
  inputSchema: {
    type: "object",
    properties: {
      mode: {
        type: "string",
        enum: ["cascade", "gcp", "demo", "default"],
        description:
          "Omit to just check. 'cascade', 'gcp', or 'demo' to switch; 'default' clears the " +
          "override and reverts to the deployed configuration.",
      },
      minutes: {
        type: "number",
        description:
          "How long demo mode stays on before reverting by itself. Defaults to 30. Only " +
          "meaningful with mode 'demo'. Pass 0 to leave it on until switched back by hand, " +
          "which should be rare: it is the expensive architecture.",
      },
    },
  },
};

/**
 * How long a fused-architecture switch stays on before reverting by itself.
 * Defaults to 30 minutes rather than staying on: a demo that outlives the
 * meeting is the expensive failure, and one that ends a few minutes early can
 * always be switched on again. 0 means no deadline, for the rare case that
 * genuinely wants one.
 */
function ttlFrom(raw: unknown): number {
  if (raw === undefined || raw === null) return 30;
  return Math.max(0, Math.floor(Number(raw) || 0));
}

const CHECK_CALLS_TOOL: ToolDef = {
  name: "check_calls",
  description:
    "Who has phoned the number, and what happened on each call. Use this whenever Lewis asks " +
    "whether anyone has called, who called, what they wanted, or whether a call led anywhere. " +
    "Returns each call's time, the caller's number, how long they stayed on, whether they have " +
    "called before, what Elle noted during that call, and any booking that came out of it.",
  inputSchema: {
    type: "object",
    properties: {
      hours: {
        type: "number",
        description:
          "How far back to look, in hours. Defaults to 24. Use 168 for 'this week', and a larger " +
          "number when Lewis asks about a longer stretch.",
      },
      limit: { type: "number", description: "Most calls to list. Defaults to 10." },
    },
  },
};

const TG_TOOLS: ToolDef[] = [SEND_OUTREACH_TOOL, CHECK_OUTREACH_TOOL, CHECK_CALLS_TOOL, VOICE_MODE_TOOL];

/** Normalize to E.164. Accepts +<country><number> or bare US 10/11-digit. */
function normalizePhone(raw: string): string | null {
  const digits = raw.replace(/[^\d+]/g, "");
  if (/^\+\d{8,15}$/.test(digits)) return digits;
  const bare = digits.replace(/\D/g, "");
  if (bare.length === 10) return `+1${bare}`;
  if (bare.length === 11 && bare.startsWith("1")) return `+${bare}`;
  return null;
}

const TERMS_URL = "https://lewiswcampbell.com/sms-terms.html";

function outreachSms(name: string, topic: string | null): string {
  const about = topic ? ` about ${topic}` : "";
  return (
    `Hi ${name}, this is ${config.assistantName}, ${config.ownerName}'s AI assistant. ` +
    `${config.ownerName} shared your number and asked me to reach out${about} — I can find a ` +
    `time for a quick intro call with him, or you can call this number to talk it through. ` +
    `Msg & data rates may apply. Reply STOP to opt out, HELP for help. Terms: ${TERMS_URL}`
  );
}

/** SQLite writes "YYYY-MM-DD HH:MM:SS" in UTC with no zone marker on it. */
function parseDbTime(value: string): Date {
  return new Date(value.replace(" ", "T") + "Z");
}

/** "just now", "12 min ago", "3 hours ago", "2 days ago". */
function ago(from: Date, now: Date): string {
  const mins = Math.max(0, Math.round((now.getTime() - from.getTime()) / 60000));
  if (mins < 2) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

/** Wall-clock time in the owner's timezone, so he doesn't convert from UTC. */
function localTime(d: Date): string {
  try {
    return d.toLocaleString("en-US", {
      timeZone: config.booking.timezone,
      weekday: "short",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
  }
}

/** A booking's start, with its date, in the owner's timezone rather than UTC. */
function localDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  try {
    return d.toLocaleString("en-US", {
      timeZone: config.booking.timezone,
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function duration(startedAt: string, endedAt: string | null): string {
  if (!endedAt) return "still on the call";
  const secs = Math.max(
    0,
    Math.round((parseDbTime(endedAt).getTime() - parseDbTime(startedAt).getTime()) / 1000)
  );
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, "0")}s`;
}

/**
 * The owner's "has anyone called?" answer, as text for the model to relay.
 *
 * Exported so the offline verify can assert the shape of it against a seeded
 * database without a network, a phone, or the intake platform.
 *
 * Two deliberate choices. Facts are read PER CALL rather than per number,
 * because notes from an earlier conversation on a shared phone must not be
 * reported as what this caller said. And an empty result names the retention
 * limit rather than flatly claiming nobody rang: the call log lives in SQLite
 * on the instance, so a deploy or a scale-down takes the history with it, and
 * "no calls" and "no memory of calls" would otherwise be indistinguishable.
 */
export function summarizeRecentCalls(
  hours: number,
  limit: number,
  now: Date = new Date(),
  uptimeSeconds: number = process.uptime()
): string {
  const calls = db.getCallsSince(hours, limit);
  const window = hours === 24 ? "the last 24 hours" : `the last ${hours} hours`;

  if (!calls.length) {
    const uptimeHours = uptimeSeconds / 3600;
    const caveat =
      uptimeHours < hours
        ? ` Note: this instance has only been up ${
            uptimeHours < 1 ? `${Math.round(uptimeSeconds / 60)} minutes` : `${uptimeHours.toFixed(1)} hours`
          }, and the call log doesn't survive a redeploy, so anything before that is not on record here.`
        : "";
    return `No calls in ${window}.${caveat}`;
  }

  const lines = calls.map((c) => {
    const started = parseDbTime(c.started_at);
    const priorCalls = db.countRecentCalls(c.phone_number, 24 * 30) - 1;
    const parts = [
      `${ago(started, now)} (${localTime(started)}) from ${c.phone_number}`,
      duration(c.started_at, c.ended_at),
    ];
    if (priorCalls > 0) parts.push(`${priorCalls} earlier call${priorCalls === 1 ? "" : "s"} in the last 30 days`);

    const facts = db.getFactsForCall(c.call_sid);
    if (facts.length) {
      parts.push("noted: " + facts.map((f) => `${f.fact} (${f.category})`).join("; "));
    }

    const booked = db
      .getBookingsForPhone(c.phone_number)
      .filter((b) => b.call_sid === c.call_sid);
    if (booked.length) {
      parts.push(
        "BOOKED: " +
          booked.map((b) => `${b.name}, ${localDateTime(b.start_at)} (${b.status})`).join(", ")
      );
    } else if (!facts.length) {
      // Nothing was learned and nothing was booked. Say so rather than leaving
      // a bare line that reads like data is missing.
      parts.push("nothing noted, no booking");
    }

    return "- " + parts.join(" · ");
  });

  const bookedCount = calls.filter(
    (c) => db.getBookingsForPhone(c.phone_number).filter((b) => b.call_sid === c.call_sid).length
  ).length;
  const header =
    `${calls.length} call${calls.length === 1 ? "" : "s"} in ${window}` +
    (bookedCount ? `, ${bookedCount} of them booked` : ", none booked") +
    ":";

  return [header, ...lines].join("\n");
}

async function runTgTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case "send_outreach": {
      const personName = String(args.name ?? "").trim();
      const topic = args.topic ? String(args.topic).trim() : null;
      if (!personName) return "Error: I need the person's name before sending.";
      const phone = normalizePhone(String(args.phone ?? ""));
      if (!phone) {
        return "Error: that phone number doesn't look valid. Ask Lewis to re-send it (E.164 like +15551234567, or a US 10-digit number).";
      }
      const consent = db.getSmsConsent(phone);
      if (consent?.status === "opted_out") {
        return `Error: ${phone} has opted out of texts (STOP). I can't message them — Lewis would need to reach out himself.`;
      }
      const body = outreachSms(personName, topic);
      let sent: boolean;
      try {
        sent = await sendSms(phone, body);
      } catch (e) {
        console.error("[telegram] outreach send failed:", e);
        return "Error: Twilio rejected the send. Tell Lewis the text did not go out and the number may be a landline or unreachable.";
      }
      if (!sent) {
        return "Error: outbound SMS isn't configured on the server (TWILIO_ACCOUNT_SID / TWILIO_FROM_NUMBER). Nothing was sent.";
      }
      db.saveOutreach(phone, personName, topic, body);
      console.log(`[telegram] outreach sent to ${personName} (${phone})${topic ? ` re: ${topic}` : ""}`);
      return `Sent. The text to ${personName} (${phone}) read: "${body}"`;
    }

    case "check_calls": {
      const hours = Math.min(24 * 90, Math.max(1, Math.floor(Number(args.hours) || 24)));
      const limit = Math.min(50, Math.max(1, Math.floor(Number(args.limit) || 10)));
      return summarizeRecentCalls(hours, limit);
    }

    case "check_outreach": {
      const rows = db.getRecentOutreach(10);
      if (!rows.length) return "No outreach on record yet.";
      return rows
        .map((o) => {
          let line = `${o.name} (${o.phone_number}) — ${o.status}, sent ${o.created_at} UTC`;
          if (o.topic) line += `, topic: ${o.topic}`;
          const bookings = db.getBookingsForPhone(o.phone_number);
          if (bookings.length) {
            line +=
              ". Bookings: " +
              bookings.map((b) => `${b.start_at} (${b.status})`).join("; ");
          }
          const consent = db.getSmsConsent(o.phone_number);
          if (consent?.status === "opted_out") line += ". NOTE: has opted out (STOP).";
          return line;
        })
        .join("\n");
    }

    case "voice_mode": {
      const label = (p: string) =>
        p === "deepgram"
          ? "cascade (production)"
          : p === "gemini"
            ? "gcp (duplex model on Vertex AI)"
            : "demo (duplex model on OpenAI)";
      const mode = args.mode ? String(args.mode) : "";
      if (!mode) {
        const left = voiceProviderMinutesLeft();
        return (
          `Current voice architecture: ${label(currentVoiceProvider())}` +
          `${voiceProviderOverridden() ? " — set by owner override" : " — deployed default"}` +
          `${left ? `, reverting to the cascade in ${left} min` : ""}.`
        );
      }
      if (mode === "default") {
        setVoiceProvider(null);
        return `Override cleared. Next call uses the deployed default: ${label(currentVoiceProvider())}.`;
      }
      // Both fused architectures share the deadline logic; only the credential
      // check and the wording differ.
      if (mode === "gcp") {
        if (!config.vertex.projectId) {
          return (
            "Error: the GCP duplex model needs VERTEX_PROJECT_ID set so it knows which project " +
            "to bill and authenticate against. Nothing was switched; calls stay on the cascade."
          );
        }
        const ttl = ttlFrom(args.minutes);
        setVoiceProvider("gemini", ttl);
        return (
          "Switched: the NEXT call answers with the duplex model on Vertex AI. No extra vendor, " +
          "same BAA, but several times the cascade's cost per minute. " +
          (ttl
            ? `It reverts to the cascade by itself in ${ttl} minutes. `
            : "No deadline set, so it stays on until switched back. ") +
          "Say 'voice mode cascade' here to switch back sooner."
        );
      }
      if (mode === "demo") {
        if (!config.realtime.apiKey) {
          return (
            "Error: demo mode needs OPENAI_REALTIME_API_KEY on the server and it isn't set. " +
            "Nothing was switched; calls stay on the cascade."
          );
        }
        const ttl = ttlFrom(args.minutes);
        setVoiceProvider("openai", ttl);
        return (
          "Switched: the NEXT call answers with the demo interaction model (~5-10x cost per minute). " +
          (ttl
            ? `It reverts to the cascade by itself in ${ttl} minutes. `
            : "No deadline set, so it stays on until switched back. ") +
          "Say 'voice mode cascade' here to switch back sooner."
        );
      }
      if (mode === "cascade") {
        setVoiceProvider("deepgram");
        return "Switched: the NEXT call answers with the production cascade.";
      }
      return "Error: mode must be cascade, demo, or default.";
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

// --- persona -----------------------------------------------------------------

function tgPersona(): string {
  const owner = config.ownerName;
  const elle = config.assistantName;
  return (
    `You are ${elle}, ${owner}'s AI assistant, chatting with ${owner} himself on Telegram. This ` +
    `is his private channel for directing your outreach: he gives you a name and phone number, ` +
    `and you text that person from his Twilio number to set up an intro call.\n\n` +
    `Rules:\n` +
    `- You're talking to your own boss: be natural, concise, and a little warm. No disclosures ` +
    `or compliance language here — that goes in the SMS, not to him.\n` +
    `- Before send_outreach, read back the name, number, and topic in one line and get a clear ` +
    `yes. If the topic is missing, ask for a few words about what the call is regarding — or ` +
    `send without one if he says skip it.\n` +
    `- After sending, confirm what went out. If a send fails, say exactly why.\n` +
    `- When he asks how outreach is going, call check_outreach and summarize in plain sentences.\n` +
    `- He can switch the phone's voice architecture with voice_mode: "demo mode on" means ` +
    `mode=demo, "back to normal"/"cascade" means mode=cascade. Confirm what the next call will use.\n` +
    `- If someone has opted out (STOP), you cannot text them — say so plainly.\n` +
    `- Anything else he asks (status, ideas, quick questions) just answer helpfully; only the ` +
    `two tools touch the outside world.`
  );
}

function tgContext(): string {
  return (
    `Right now it is ${new Date().toISOString()} (UTC). ${config.ownerName}'s timezone is ` +
    `${config.booking.timezone}. Replies from people you text arrive on the SMS channel, where ` +
    `you handle scheduling with them directly — Lewis only needs to hand you the contact.`
  );
}

// --- webhook -----------------------------------------------------------------

export const telegramRouter = Router();

telegramRouter.post("/telegram", (req: Request, res: Response) => {
  // Ack immediately — Telegram retries slow webhooks, which would double-send.
  res.status(200).send("ok");

  const update = (req.body ?? {}) as Record<string, unknown>;
  void handleUpdate(req, update).catch((e) => console.error("[telegram] update failed:", e));
});

async function handleUpdate(req: Request, update: Record<string, unknown>): Promise<void> {
  if (!config.telegram.botToken) return;

  if (config.telegram.secret) {
    const header = req.header("X-Telegram-Bot-Api-Secret-Token") ?? "";
    const a = Buffer.from(header);
    const b = Buffer.from(config.telegram.secret);
    if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
      console.warn("[telegram] dropped update with bad secret token");
      return;
    }
  }

  const message = update.message as
    | { text?: string; chat?: { id: number }; from?: { id: number; first_name?: string } }
    | undefined;
  const text = message?.text?.trim();
  const chatId = message?.chat?.id;
  const senderId = message?.from?.id;
  if (!text || chatId == null || senderId == null) return;

  // Setup mode: no owner configured yet — tell the sender their id and stop.
  if (!config.telegram.ownerId) {
    await tgSend(
      chatId,
      `Setup: your Telegram user id is ${senderId}. Set TELEGRAM_OWNER_ID=${senderId} on the ` +
        `server and redeploy, and I'll start listening to you here.`
    );
    return;
  }
  if (String(senderId) !== config.telegram.ownerId) {
    console.warn(`[telegram] ignored message from non-owner ${senderId}`);
    return;
  }

  const thread = getOwnerThread();
  thread.messages.push({ role: "user", text });
  thread.lastAt = Date.now();
  db.saveTranscript("telegram:owner", "caller", text);
  console.log(`[telegram] lewis: "${text}"`);

  let replyText = "Sorry, something went wrong on my end — try that again.";
  try {
    const working: NeutralMessage[] = [...thread.messages];
    let resp = await provider.chat({
      systemStable: tgPersona(),
      systemDynamic: tgContext(),
      messages: working,
      tools: TG_TOOLS,
      maxTokens: TG_MAX_TOKENS,
    });
    for (let hop = 0; hop < 5 && resp.toolCalls.length > 0; hop++) {
      working.push({ role: "assistant", text: resp.text, toolCalls: resp.toolCalls, raw: resp.raw });
      const results: ToolResultMsg[] = [];
      for (const call of resp.toolCalls) {
        let output: string;
        try {
          output = await runTgTool(call.name, call.input);
        } catch (e) {
          console.error(`[telegram] tool ${call.name} failed:`, e);
          output = "Error: that action failed unexpectedly. Nothing was sent.";
        }
        results.push({ id: call.id, content: output, isError: output.startsWith("Error") });
      }
      working.push({ role: "tool_results", results });
      resp = await provider.chat({
        systemStable: tgPersona(),
        systemDynamic: tgContext(),
        messages: working,
        tools: TG_TOOLS,
        maxTokens: TG_MAX_TOKENS,
      });
    }
    if (resp.text.trim()) replyText = resp.text.trim();
  } catch (e) {
    console.error("[telegram] turn failed:", e);
  }

  thread.messages.push({ role: "assistant", text: replyText, toolCalls: [] });
  if (thread.messages.length > THREAD_MAX_MESSAGES) {
    thread.messages = thread.messages.slice(-THREAD_MAX_MESSAGES);
  }
  db.saveTranscript("telegram:owner", "assistant", replyText);
  console.log(`[telegram] elle: "${replyText}"`);
  await tgSend(chatId, replyText);
}
