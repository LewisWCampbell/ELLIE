import { Router, Request, Response, NextFunction } from "express";
import crypto from "crypto";
import twilio from "twilio";
import { config } from "../config";
import { assessCaller, blockedMessage } from "../callerPolicy";
import { liveCallCount } from "../sessions";

/**
 * Signed, short-lived token passed to the /media WebSocket via <Stream>
 * custom parameters. Twilio doesn't sign WebSocket connects, so without this
 * anyone could open /media, send a fake "start" frame, and run the paid
 * STT/LLM/TTS pipeline. HMAC over callSid + issue time, keyed with the Twilio
 * auth token (already secret, already required for webhook validation).
 */
export function mintMediaToken(callSid: string, ts = Date.now()): string {
  const sig = crypto
    .createHmac("sha256", config.twilioAuthToken)
    .update(`${callSid}|${ts}`)
    .digest("hex");
  return `${ts}.${sig}`;
}

const MEDIA_TOKEN_MAX_AGE_MS = 5 * 60 * 1000;

/**
 * Verify a /media stream token. The no-token branch is reachable only under
 * ALLOW_UNSIGNED_TWILIO=1: boot refuses to start without an auth token
 * otherwise (see src/index.ts), so production always takes the HMAC path.
 */
export function verifyMediaToken(callSid: string, token: string): boolean {
  if (!config.twilioAuthToken) return true;
  const [tsStr, sig] = String(token ?? "").split(".");
  const ts = Number(tsStr);
  if (!Number.isFinite(ts) || Math.abs(Date.now() - ts) > MEDIA_TOKEN_MAX_AGE_MS) return false;
  const expected = crypto
    .createHmac("sha256", config.twilioAuthToken)
    .update(`${callSid}|${ts}`)
    .digest("hex");
  const a = Buffer.from(String(sig ?? ""));
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

/**
 * Twilio voice webhook for the realtime (barge-in) path. Instead of the
 * Gather/Say turn loop, it connects the call's audio to our /media WebSocket
 * with <Connect><Stream>, which the Twilio↔OpenAI bridge then drives.
 */

const VoiceResponse = twilio.twiml.VoiceResponse;
export const realtimeRouter = Router();

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

realtimeRouter.post("/voice-realtime", validateTwilio, (req: Request, res: Response) => {
  const from = String(req.body.From ?? "unknown");
  const callSid = String(req.body.CallSid ?? "");
  const { tier } = assessCaller(from);

  const twiml = new VoiceResponse();

  // Blocked caller: say a short line and hang up — never open the media stream,
  // so no Deepgram/LLM/TTS cost is incurred at all.
  if (tier === "blocked") {
    twiml.say({ voice: config.ttsVoice as never }, blockedMessage());
    twiml.hangup();
    res.type("text/xml").send(twiml.toString());
    return;
  }

  // At capacity: say so and invite a text, rather than opening a media stream
  // this instance cannot serve. The caller previously heard silence here and
  // hung up believing the number was broken, which loses the lead and looks
  // like a fault. Checked before the stream opens, so no paid pipeline starts.
  if (liveCallCount() >= config.maxConcurrentCalls) {
    console.warn(`[capacity] refused ${from}: ${liveCallCount()} calls in progress`);
    twiml.say(
      { voice: config.ttsVoice as never },
      `Sorry, I'm on another call right now. Please send me a text at this same number and ` +
        `I'll get right back to you, or try again in a few minutes.`
    );
    twiml.hangup();
    res.type("text/xml").send(twiml.toString());
    return;
  }

  const wsBase = config.publicUrl.replace(/^http/, "ws"); // https→wss, http→ws
  const connect = twiml.connect();
  const stream = connect.stream({ url: `${wsBase}/media` });
  // Pass caller identity + tier through to the media stream.
  stream.parameter({ name: "from", value: from });
  stream.parameter({ name: "callSid", value: callSid });
  stream.parameter({ name: "tier", value: tier });
  stream.parameter({ name: "mediaToken", value: mintMediaToken(callSid) });

  res.type("text/xml").send(twiml.toString());
});
