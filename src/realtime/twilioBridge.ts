import WebSocket, { WebSocketServer } from "ws";
import { config } from "../config";
import * as db from "../db";
import { openAIRealtimeFactory } from "./openaiRealtime";
import { RealtimeFactory, RealtimeModel, RealtimeTool } from "./types";
import { personaCore, personaContext, SEARCH_RESUME, SPANISH_INVITE, spanishEnabled } from "../persona";
import { searchKnowledge, formatRetrieved } from "../knowledge";
import { CallSession, classifyShowUrl, createSession, endSession, isPaired, pushWebEvent, pushFactsSnapshot } from "../sessions";
import { notifyCallEnded } from "../notify";
import { verifyMediaToken } from "./route";
import { CallerTier, limitedNote } from "../callerPolicy";
import { CallRecorder } from "../recording/recorder";
import { BlobStore, createBlobStore } from "../recording/blob";
import { BOOK_INTRO_CALL, handleBookIntroCall } from "../booking/booking";
import { CHECK_AVAILABILITY, handleCheckAvailability } from "../booking/availability";
import { sendConsentConfirmation } from "../sms";
import { logQuestion } from "../gaps";

// One store for the process; null when recording is disabled/unconfigured.
const blobStore: BlobStore | null = createBlobStore();

/**
 * Bridges a Twilio Media Stream (bidirectional call audio over WebSocket) to
 * the OpenAI Realtime API. One Bridge instance per phone call.
 *
 * Flow:  Twilio caller audio ──▶ OpenAI Realtime ──▶ spoken reply ──▶ Twilio
 *        On barge-in (caller talks over the bot) we flush Twilio's playback
 *        buffer and truncate the model's context so it adapts to the interruption.
 */

const SEARCH_TOOL: RealtimeTool = {
  name: SEARCH_RESUME.name,
  description: SEARCH_RESUME.description,
  parameters: SEARCH_RESUME.parameters,
};

const SAVE_FACT_TOOL: RealtimeTool = {
  name: "save_user_fact",
  description:
    "Record a concise fact about the caller (their name, company, the role they're " +
    "hiring for, why they called, preferences). Call this whenever the caller reveals " +
    "something meaningful about themselves. Do it quietly — don't announce it.",
  parameters: {
    type: "object",
    properties: {
      category: { type: "string", description: "Short label, e.g. name, company, role, reason_for_calling." },
      fact: { type: "string", description: "A standalone sentence, e.g. 'Their name is Priya.'" },
    },
    required: ["category", "fact"],
  },
};

const SHOW_WORK_TOOL: RealtimeTool = {
  name: "show_work",
  description:
    "Display a piece of Lewis's work in the caller's paired web browser (only works after they pair " +
    "using the pairing code). Prefer the LIVE, interactive URL for a project when the visuals list " +
    "has one — the caller can actually use it while you talk — and fall back to a screenshot when " +
    "it doesn't. After showing something, ask a question about it. HTTPS URL from the visuals list only.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "Publicly accessible HTTPS URL: a live page or an image." },
      caption: { type: "string", description: "Short caption naming what they're looking at." },
    },
    required: ["url", "caption"],
  },
};

const BOOK_TOOL: RealtimeTool = {
  name: BOOK_INTRO_CALL.name,
  description: BOOK_INTRO_CALL.description,
  parameters: BOOK_INTRO_CALL.parameters,
};

const AVAILABILITY_TOOL: RealtimeTool = {
  name: CHECK_AVAILABILITY.name,
  description: CHECK_AVAILABILITY.description,
  parameters: CHECK_AVAILABILITY.parameters,
};

const SHOW_CHOICES_TOOL: RealtimeTool = {
  name: "show_choices",
  description:
    "Display a question with clickable options in the caller's paired browser (only after they pair).",
  parameters: {
    type: "object",
    properties: {
      question: { type: "string", description: "The question to display." },
      options: { type: "array", items: { type: "string" }, description: "2–6 short options." },
    },
    required: ["question", "options"],
  },
};

// Inline grounding: the whole (small) knowledge base is embedded in the prompt
// (inside personaCore) so the phone brain answers in one model call, with no
// search_resume round-trip. The static/dynamic split matters for cost: the
// static part is identical for every call, so the prompt cache shares it
// across ALL calls; the per-call part (facts, pairing code, times) rides in
// instructionsDynamic after it, where it can't break the shared prefix.

/**
 * The opening line, spoken the instant the call connects.
 *
 * Exported so it can be asserted on offline. This text used to be unreachable
 * from any test: thirteen suites passed while the greeting stumbled over the
 * owner's name for weeks, because every suite stubbed the audio away and no
 * suite could even see the string. See src/greeting.verify.ts.
 */
/**
 * The opening line, spoken the instant the call connects.
 *
 * Exported so it can be asserted on offline. This text used to be unreachable
 * from any test: thirteen suites passed while the greeting stumbled over the
 * owner's name for weeks, because every one of them stubbed the audio away and
 * none could even see the string. See src/greeting.verify.ts.
 *
 * Worded for the ear, not the eye. Aura leans on commas, so the earlier "this
 * is Elle, Lewis Campbell's AI assistant" put a beat between two names and the
 * intro landed as "Elle... Lewis... Campbell's". No name may sit directly
 * against a comma pause, and the full name is said exactly once: repeats are
 * first-name only, which is how a person talks.
 *
 * And kept SHORT. The greeting cannot be interrupted, so every word of it is
 * dead air the caller has to sit through; the previous version ran past thirty
 * seconds, which is long enough to hang up on. What was cut is not lost: the
 * line about texting to change a booking now goes out when a booking is
 * actually made, which is both shorter here and more useful there.
 */
export function buildGreeting(recordingOn: boolean): string {
  let greeting =
    `Hi there! I'm ${config.assistantName}, the AI assistant for ${config.ownerName}. You can ask me ` +
    `about ${config.ownerFirstName}'s work, book a short intro call, or pair your browser at ` +
    `${config.siteUrl.replace(/^https?:\/\//, "")} to see his projects while we talk. ` +
    `So, who am I speaking with?`;
  if (recordingOn) {
    greeting = `Just so you know, this call is recorded. ${greeting}`;
  }
  return greeting;
}

export class Bridge {
  private rs: RealtimeModel | null = null;
  private session: CallSession | null = null;
  private streamSid = "";
  private callStartedAt = 0;
  private callReported = false;
  private from = "unknown";
  private callSid = "";
  private tier: CallerTier = "normal";
  private maxDurationTimer: NodeJS.Timeout | null = null;
  /** Fires if the caller never says anything — see silentHangupSeconds. */
  private silenceTimer: NodeJS.Timeout | null = null;
  private recorder: CallRecorder | null = null;
  private recordingFinalized = false;
  /** Approx. ms of bot audio actually sent to Twilio since the current utterance began. */
  private playedMs = 0;
  /** Wall-clock time Elle's current reply started speaking, for the barge-in grace window. */
  private botUtteranceStart = 0;
  /**
   * Greeting lock: the opening message always plays to the end. Cumulative ms
   * of audio enqueued/sent over the whole call; greetingEndMs is stamped (via
   * onGreetingSpoken) with the enqueued total once the greeting is fully
   * generated. Barge-in stays disabled until that much audio has been SENT.
   */
  private totalEnqueuedMs = 0;
  private totalSentMs = 0;
  private greetingEndMs: number | null = null;
  private greetingGate = false;
  /** Outbound 20 ms μ-law frames awaiting a paced send to Twilio (base64). */
  private audioQueue: string[] = [];
  /** Drains audioQueue toward Twilio on a drift-corrected 20 ms schedule. */
  private paceTimer: NodeJS.Timeout | null = null;
  /** Wall-clock start of the current playout batch, for drift-corrected pacing. */
  private paceStartMs = 0;
  /** Frames already sent in the current batch (compared against the ideal count). */
  private paceSent = 0;

  constructor(private twilio: WebSocket, private createModel: RealtimeFactory) {
    twilio.on("message", (d) => this.onTwilioMessage(d));
    twilio.on("close", () => {
      if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
      if (this.silenceTimer) clearTimeout(this.silenceTimer);
      this.clearAudio();
      this.rs?.close();
      void this.finalizeRecording();
      this.reportCall();
      if (this.callSid) endSession(this.callSid);
    });
  }

  /**
   * Report the call to the owner, exactly once.
   *
   * Both close paths reach here (Twilio's socket closing, and the "stop" frame),
   * and on a normal hangup both fire, so the guard is the point. Facts and turns
   * are read back from the database rather than tracked in memory: it is the
   * same source the owner's other views use, so the notification cannot claim
   * something the record does not contain.
   */
  private reportCall(): void {
    if (this.callReported || !this.callSid) return;
    this.callReported = true;
    try {
      const facts = db.getFactsForCall(this.callSid);
      const turns = db.getTranscripts(this.callSid).filter((t) => t.role === "caller").length;
      const seconds = this.callStartedAt ? Math.round((Date.now() - this.callStartedAt) / 1000) : 0;
      notifyCallEnded({
        phoneNumber: this.from || "unknown",
        seconds,
        turns,
        facts: facts.map((f) => ({ category: f.category, fact: f.fact })),
        booked: db.getBookingsForCall(this.callSid).length > 0,
      });
    } catch (e) {
      console.error("[notify] call report failed:", e);
    }
  }

  private sendTwilio(obj: unknown): void {
    if (this.twilio.readyState === WebSocket.OPEN) this.twilio.send(JSON.stringify(obj));
  }

  /**
   * Queue one 20 ms μ-law frame for playback. Streaming TTS delivers audio in
   * fast bursts; sending it straight through makes Twilio's playback stutter, so
   * we emit the first frame at once (low latency) and then pace the rest at one
   * frame per 20 ms — a steady real-time stream Twilio plays smoothly.
   */
  private enqueueAudio(b64: string): void {
    this.totalEnqueuedMs += Buffer.from(b64, "base64").length / 8;
    this.audioQueue.push(b64);
    if (!this.paceTimer) {
      this.paceStartMs = Date.now();
      this.paceSent = 0;
      this.flushDueFrames();
      // Tick faster than the frame period and catch up to wall-clock, so timer
      // jitter can't accumulate into a starved (slurred) tail.
      this.paceTimer = setInterval(() => this.flushDueFrames(), 10);
      this.paceTimer.unref?.();
    }
  }

  private flushDueFrames(): void {
    // How many 20 ms frames should have been sent by now, from the batch start.
    const due = Math.floor((Date.now() - this.paceStartMs) / 20) + 1;
    while (this.paceSent < due && this.audioQueue.length > 0) {
      const b64 = this.audioQueue.shift() as string;
      this.sendTwilio({ event: "media", streamSid: this.streamSid, media: { payload: b64 } });
      // G.711 μ-law @ 8 kHz = 8 bytes per millisecond of audio.
      const ms = Buffer.from(b64, "base64").length / 8;
      this.playedMs += Math.round(ms);
      this.totalSentMs += ms;
      this.paceSent++;
    }
    if (this.audioQueue.length === 0) {
      if (this.paceTimer) clearInterval(this.paceTimer);
      this.paceTimer = null;
    }
  }

  /** Drop all queued audio immediately (barge-in / hang-up) and stop the pacer. */
  private clearAudio(): void {
    this.audioQueue = [];
    if (this.paceTimer) clearInterval(this.paceTimer);
    this.paceTimer = null;
    this.paceSent = 0;
  }

  /**
   * Render the captured audio to a stereo WAV and upload it plus a JSON sidecar
   * (metadata + transcript) to blob storage. Fire-and-forget on hang-up; runs at
   * most once per call and never throws into the socket teardown.
   */
  private async finalizeRecording(): Promise<void> {
    if (this.recordingFinalized || !blobStore || !this.recorder) return;
    this.recordingFinalized = true;
    const rec = this.recorder;
    if (rec.isEmpty) return;
    try {
      const date = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
      const base = `${config.recording.prefix}/${date}/${this.callSid}`;
      const uri = await blobStore.put(`${base}.wav`, rec.toWav(), "audio/wav");
      const meta = {
        callSid: this.callSid,
        from: this.from,
        tier: this.tier,
        durationMs: rec.durationMs,
        recordedAt: new Date().toISOString(),
        transcript: db.getTranscripts(this.callSid),
      };
      await blobStore.put(`${base}.json`, Buffer.from(JSON.stringify(meta, null, 2)), "application/json");
      console.log(`Recorded call ${this.callSid} -> ${uri} (${rec.durationMs} ms)`);
    } catch (e) {
      console.error("Recording upload failed:", e);
    }
  }

  private onTwilioMessage(data: WebSocket.RawData): void {
    let msg: Record<string, unknown> & { event?: string };
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    switch (msg.event) {
      case "start": {
        const start = (msg.start ?? {}) as { streamSid?: string; customParameters?: Record<string, string> };
        this.streamSid = start.streamSid ?? "";
        this.from = start.customParameters?.from ?? "unknown";
        this.callStartedAt = Date.now();
        this.callSid = start.customParameters?.callSid || this.streamSid;
        this.tier = (start.customParameters?.tier as CallerTier) ?? "normal";
        // Only streams our own /voice-realtime TwiML started carry a valid
        // signed token; a direct WebSocket connection with a forged start
        // frame gets dropped before any paid pipeline spins up.
        if (!verifyMediaToken(this.callSid, start.customParameters?.mediaToken ?? "")) {
          console.warn(`[call] rejected /media stream with bad token (callSid=${this.callSid})`);
          try {
            this.twilio.close();
          } catch {
            /* already closed */
          }
          return;
        }
        db.recordCallStart(this.callSid, this.from);
        console.log(`[call] start ${this.callSid} from ${this.from} tier=${this.tier}`);
        const facts = db.getFactsForCaller(this.from);
        // A session gives us a pairing code + web event fan-out, so the realtime
        // bot can push images/choices to a paired browser just like the cascaded one.
        this.session = createSession(this.callSid, this.from, facts);
        if (blobStore) this.recorder = new CallRecorder();
        this.startRealtime(facts);
        break;
      }
      case "media": {
        const media = (msg.media ?? {}) as { payload?: string };
        if (media.payload) {
          this.rs?.appendAudio(media.payload);
          this.recorder?.pushCaller(Buffer.from(media.payload, "base64"));
        }
        break;
      }
      case "stop":
        if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
        if (this.silenceTimer) clearTimeout(this.silenceTimer);
        this.clearAudio();
        console.log(`[call] end ${this.callSid} from ${this.from}`);
        db.recordCallEnd(this.callSid);
        this.rs?.close();
        void this.finalizeRecording();
        this.reportCall();
        if (this.callSid) endSession(this.callSid);
        break;
    }
  }

  private startRealtime(facts: db.Fact[]): void {
    let instructionsDynamic = personaContext(facts, this.session?.pairingCode);
    if (this.tier === "limited") {
      // Repeat caller: tell the bot to keep it short (the shorter duration
      // cap below is the backstop).
      instructionsDynamic += limitedNote();
    }
    // EVERY call gets a hard duration ceiling — a stuck line or an autodialer
    // that never hangs up must not run an open-ended bill. Limited-tier
    // callers get the shorter abuse cap. Closing the <Connect> stream ends
    // the call.
    const capSeconds =
      this.tier === "limited"
        ? Math.min(config.abuse.limitedMaxSeconds, config.abuse.maxCallSeconds)
        : config.abuse.maxCallSeconds;
    if (capSeconds > 0) {
      this.maxDurationTimer = setTimeout(() => {
        console.log(`[call] duration cap (${capSeconds}s) reached ${this.callSid} from ${this.from}`);
        try {
          this.twilio.close();
        } catch {
          /* already closed */
        }
      }, capSeconds * 1000);
      this.maxDurationTimer.unref?.();
    }
    // Silent-call guard: war-dialers and robocall probes connect and say
    // nothing, holding a paid STT stream. If the caller hasn't produced a
    // single utterance (spoken or typed via a paired browser) by the
    // deadline, hang up. Cleared on the first real utterance below.
    if (config.abuse.silentHangupSeconds > 0) {
      this.silenceTimer = setTimeout(() => {
        console.log(`[call] silent hangup (${config.abuse.silentHangupSeconds}s, no caller speech) ${this.callSid} from ${this.from}`);
        try {
          this.twilio.close();
        } catch {
          /* already closed */
        }
      }, config.abuse.silentHangupSeconds * 1000);
      this.silenceTimer.unref?.();
    }
    // Deterministic opening line so Elle speaks the instant the call connects
    // instead of waiting for the caller (no LLM/STT round-trip = no dead air).
    // If the call is recorded, the consent notice leads the greeting.
    const greeting = buildGreeting(config.recording.enabled && Boolean(blobStore));
    // Tell the model it already opened with this line, so it doesn't greet,
    // re-introduce itself, or repeat the recording notice on its first turn.
    // Greeting text is config-derived, so this stays in the STATIC block. In
    // bilingual mode the cascade closes the greeting with the Spanish invite
    // (spoken by the Spanish voice), so quote that too — otherwise the model
    // re-offers Spanish on its first turn.
    const greetingSaid = spanishEnabled() ? `${greeting} ${SPANISH_INVITE}` : greeting;
    const instructions =
      `${personaCore("voice", "inline")}\n\n` +
      `You have already opened the call by saying: "${greetingSaid}" Do not greet again, ` +
      `re-introduce yourself, or repeat that notice — reply directly to what the caller says next.`;

    this.rs = this.createModel(
      {
        instructions,
        instructionsDynamic,
        voice: config.realtime.voice,
        // No SEARCH_TOOL: the knowledge base is embedded inline (buildInstructions),
        // so she answers without a retrieval round-trip. Keeps latency down.
        tools: [SAVE_FACT_TOOL, AVAILABILITY_TOOL, BOOK_TOOL, SHOW_WORK_TOOL, SHOW_CHOICES_TOOL],
      },
      {
        onAudioDelta: (b64) => {
          this.recorder?.pushBot(Buffer.from(b64, "base64"));
          this.enqueueAudio(b64); // paced send to Twilio (smooths choppy playback)
        },
        onSpeechStarted: () => {
          // The opening greeting always runs to completion: barge-in stays
          // disabled until every frame of it has been sent to the caller.
          // (Their words still land — the utterance queues as the next turn.)
          if (this.greetingGate) {
            if (this.greetingEndMs === null || this.totalSentMs < this.greetingEndMs) return;
            this.greetingGate = false;
          }
          // Only treat caller speech as an interruption while Elle is actually
          // TALKING (audio queued/playing). While she's thinking, canceling
          // would silently kill the reply being generated — the caller hears
          // dead air and has to say "are you there?". Their words still land:
          // the STT utterance queues as the next turn.
          if (this.audioQueue.length === 0 && !this.paceTimer) return;
          // Grace window: ignore barge-in in the opening moments of Elle's own
          // reply, so line noise or an echo of her voice can't cut her off the
          // instant she starts. After the window, genuine interruptions work.
          if (Date.now() - this.botUtteranceStart < config.deepgram.bargeInGraceMs) return;
          // BARGE-IN. Caller talked over the bot:
          //  1) flush audio already queued (ours + Twilio's) so she goes quiet at once,
          //  2) truncate the model's memory of its reply to what was actually heard,
          //  3) cancel the in-progress response.
          this.sendTwilio({ event: "clear", streamSid: this.streamSid });
          this.clearAudio();
          if (this.rs?.currentItemId) this.rs.truncate(this.rs.currentItemId, this.playedMs);
          this.rs?.cancelResponse();
          this.playedMs = 0;
        },
        onUserTranscript: (t) => {
          if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
          }
          db.saveTranscript(this.callSid, "caller", t);
          logQuestion("phone", t);
          if (this.session) pushWebEvent(this.session, { type: "transcript", role: "caller", text: t });
        },
        onGreetingSpoken: () => {
          this.greetingEndMs = this.totalEnqueuedMs;
        },
        onAssistantTranscript: (t) => {
          db.saveTranscript(this.callSid, "assistant", t);
          if (this.session) pushWebEvent(this.session, { type: "transcript", role: "assistant", text: t });
          this.playedMs = 0; // a new utterance starts fresh
          this.botUtteranceStart = Date.now(); // start the barge-in grace window
        },
        onFunctionCall: (name, args, callId) => this.handleFunctionCall(name, args, callId),
        onError: (e) => console.error("Realtime error:", e),
      }
    );
    // Typed messages from the paired browser flow straight into the call as
    // conversation turns (Elle answers them out loud).
    if (this.session && typeof this.rs.sendText === "function") {
      this.session.injectText = (text) => this.rs?.sendText?.(text);
    }

    // Speak the opening line immediately so the caller hears Elle right away.
    // The gate only arms when the model actually greets (the cascade does); a
    // model without greet() must not leave barge-in locked forever.
    this.greetingGate = typeof this.rs.greet === "function";
    this.rs.greet?.(greeting);
  }

  private handleFunctionCall(name: string, args: Record<string, unknown>, callId: string): void {
    const reply = (out: string) => this.rs?.sendFunctionResult(callId, out);
    switch (name) {
      case "search_resume": {
        const found = searchKnowledge(String(args.query ?? ""), 4);
        if (this.session) {
          for (const r of found) pushWebEvent(this.session, { type: "fact", category: "resume", fact: r.heading });
        }
        reply(formatRetrieved(found));
        break;
      }
      case "save_user_fact": {
        const category = String(args.category ?? "general");
        const fact = String(args.fact ?? "");
        if (fact) {
          db.saveFact(this.from, category, fact, this.callSid);
          if (this.session) {
            pushWebEvent(this.session, { type: "fact", category, fact });
            pushFactsSnapshot(this.session);
          }
        }
        // Verbal SMS opt-in: record consent in the ledger and send the required
        // enrollment-confirmation text (fire-and-forget; the call goes on).
        if (category === "sms_consent" && this.from && this.from !== "unknown") {
          db.setSmsConsent(this.from, "opted_in", "verbal");
          void sendConsentConfirmation(this.from).catch((e) =>
            console.error("consent confirmation SMS failed:", e)
          );
        }
        reply("Saved.");
        break;
      }
      case "check_availability": {
        // A read: no side effects, never throws (failures come back as UNKNOWN:).
        void handleCheckAvailability(args).then(reply);
        break;
      }
      case "book_intro_call": {
        // Async (calendar + DB): reply once it resolves; the model waits for it.
        void handleBookIntroCall(args, {
          fromNumber: this.from,
          callSid: this.callSid,
          notify: (s) =>
            this.session && pushWebEvent(this.session, { type: "fact", category: "booking", fact: s }),
        })
          .then(reply)
          .catch((e) => {
            console.error("book_intro_call failed:", e);
            reply("Error booking the call; offer to note their details for a follow-up instead.");
          });
        break;
      }
      case "show_work": {
        if (!this.session || !isPaired(this.session)) {
          reply("Error: the caller hasn't paired a browser yet. Give them the pairing code first.");
          break;
        }
        const url = String(args.url ?? args.image_url ?? "");
        if (!/^https:\/\//i.test(url)) {
          reply("Error: url must be a public HTTPS URL.");
          break;
        }
        const kind = classifyShowUrl(url, config.siteUrl);
        pushWebEvent(this.session, { type: "show", kind, url, caption: String(args.caption ?? "") });
        reply(
          kind === "page"
            ? "The live page is now running in the caller's browser; they can interact with it while you talk."
            : kind === "link"
              ? "That site refuses embedding, so the caller sees a card with a button to open it in a new tab."
              : "Image is now displayed in the caller's browser."
        );
        break;
      }
      case "show_choices": {
        if (!this.session || !isPaired(this.session)) {
          reply("Error: the caller hasn't paired a browser yet. Give them the pairing code first.");
          break;
        }
        const options = Array.isArray(args.options) ? (args.options as unknown[]).map(String) : [];
        if (options.length < 2) {
          reply("Error: provide at least two options.");
          break;
        }
        pushWebEvent(this.session, { type: "choices", question: String(args.question ?? ""), options });
        reply("Choices are now displayed in the caller's browser.");
        break;
      }
      default:
        reply(`Unknown tool: ${name}`);
    }
  }
}

/**
 * WebSocket server (noServer — mounted by index.ts) for Twilio Media Streams.
 * The factory getter is resolved PER CONNECTION, so an owner switching voice
 * architecture at runtime (via Telegram) takes effect on the next call while
 * in-flight calls keep the model they started with.
 */
export function createMediaWss(
  getFactory: () => RealtimeFactory = () => openAIRealtimeFactory
): WebSocketServer {
  const wss = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 });
  wss.on("connection", (ws: WebSocket) => {
    new Bridge(ws, getFactory());
  });
  return wss;
}
