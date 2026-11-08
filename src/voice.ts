/**
 * The original turn-based phone path (VOICE_MODE=cascaded): Twilio <Gather>
 * collects a whole caller utterance as text, runTurn() produces the reply, and
 * <Say> speaks it. No barge-in and a beat of latency per turn, but dead simple
 * and cheap — kept as the fallback transport and the clearest illustration of
 * the request/response shape the realtime bridge replaces. Also owns Twilio
 * webhook signature validation and the repeat-caller abuse gate for this path.
 */
import { Router, Request, Response, NextFunction } from "express";
import twilio from "twilio";
import { config } from "./config";
import * as db from "./db";
import { runTurn } from "./ai";
import {
  createSession,
  endSession,
  getSessionByCallSid,
  pushWebEvent,
} from "./sessions";
import { assessCaller, blockedMessage } from "./callerPolicy";
import { logQuestion } from "./gaps";

const VoiceResponse = twilio.twiml.VoiceResponse;

export const voiceRouter = Router();

/** Verify Twilio's request signature when an auth token is configured. */
/* Reachable without a token only under ALLOW_UNSIGNED_TWILIO=1; boot exits otherwise (src/index.ts). */
function validateTwilioRequest(req: Request, res: Response, next: NextFunction): void {
  if (!config.twilioAuthToken) {
    next();
    return;
  }
  const signature = req.header("X-Twilio-Signature") ?? "";
  const url = `${config.publicUrl}${req.originalUrl}`;
  const valid = twilio.validateRequest(config.twilioAuthToken, signature, url, req.body ?? {});
  if (!valid) {
    res.status(403).send("Invalid Twilio signature");
    return;
  }
  next();
}

function say(twiml: InstanceType<typeof VoiceResponse>, text: string): void {
  twiml.say({ voice: config.ttsVoice as never }, text);
}

/** Speak `text`, then listen for the caller's next utterance. */
function speakAndGather(text: string): string {
  const twiml = new VoiceResponse();
  const gather = twiml.gather({
    input: ["speech"],
    action: "/voice/respond",
    method: "POST",
    speechTimeout: "auto",
    speechModel: "deepgram_nova-3",
    actionOnEmptyResult: true,
  });
  gather.say({ voice: config.ttsVoice as never }, text);
  // If the gather times out with no speech, re-prompt.
  twiml.redirect({ method: "POST" }, "/voice/respond");
  return twiml.toString();
}

/** Incoming call: create the session and greet the caller. */
voiceRouter.post("/voice", validateTwilioRequest, async (req: Request, res: Response) => {
  const callSid = String(req.body.CallSid ?? "");
  const from = String(req.body.From ?? "unknown");

  // Blocked repeat caller: brief message, then hang up before any LLM work.
  if (assessCaller(from).tier === "blocked") {
    const twiml = new VoiceResponse();
    say(twiml, blockedMessage());
    twiml.hangup();
    res.type("text/xml").send(twiml.toString());
    return;
  }

  db.recordCallStart(callSid, from);
  const knownFacts = db.getFactsForCaller(from);
  const session = createSession(callSid, from, knownFacts);

  try {
    const greeting = await runTurn(
      session,
      "<call_started>The phone call has just connected. Greet the caller.</call_started>"
    );
    db.saveTranscript(callSid, "assistant", greeting);
    pushWebEvent(session, { type: "transcript", role: "assistant", text: greeting });
    res.type("text/xml").send(speakAndGather(greeting));
  } catch (err) {
    console.error("Error on call start:", err);
    const twiml = new VoiceResponse();
    say(twiml, "Sorry, I'm having trouble starting up right now. Please call back in a moment.");
    twiml.hangup();
    res.type("text/xml").send(twiml.toString());
  }
});

/** Each conversational turn: Twilio posts the caller's transcribed speech here. */
voiceRouter.post("/voice/respond", validateTwilioRequest, async (req: Request, res: Response) => {
  const callSid = String(req.body.CallSid ?? "");
  const session = getSessionByCallSid(callSid);

  if (!session) {
    const twiml = new VoiceResponse();
    say(twiml, "Sorry, something went wrong with our call. Please phone back.");
    twiml.hangup();
    res.type("text/xml").send(twiml.toString());
    return;
  }

  const speech = String(req.body.SpeechResult ?? "").trim();
  const callerText =
    speech ||
    "<no_speech>The caller didn't say anything. Gently prompt them again, or ask if they're still there.</no_speech>";

  if (speech) {
    db.saveTranscript(callSid, "caller", speech);
    logQuestion("phone", speech);
    pushWebEvent(session, { type: "transcript", role: "caller", text: speech });
  }

  try {
    const reply = await runTurn(session, callerText);
    db.saveTranscript(callSid, "assistant", reply);
    pushWebEvent(session, { type: "transcript", role: "assistant", text: reply });

    if (session.shouldHangup) {
      const twiml = new VoiceResponse();
      say(twiml, reply);
      twiml.hangup();
      res.type("text/xml").send(twiml.toString());
      return;
    }

    res.type("text/xml").send(speakAndGather(reply));
  } catch (err) {
    console.error("Error during turn:", err);
    res
      .type("text/xml")
      .send(speakAndGather("Sorry, I had a hiccup there. Could you say that again?"));
  }
});

/** Twilio status callback: clean up when the call ends for any reason. */
voiceRouter.post("/voice/status", validateTwilioRequest, (req: Request, res: Response) => {
  const callSid = String(req.body.CallSid ?? "");
  const status = String(req.body.CallStatus ?? "");
  if (["completed", "busy", "failed", "no-answer", "canceled"].includes(status)) {
    db.recordCallEnd(callSid);
    endSession(callSid);
  }
  res.sendStatus(204);
});
