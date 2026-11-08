import { config } from "../config";
import { provider as defaultLlm } from "../llm/provider";
import { ChatResponse, LlmProvider, NeutralMessage, ToolDef, ToolResultMsg } from "../llm/types";
import { SPANISH_INVITE, spanishEnabled } from "../persona";
import { RealtimeFactory, RealtimeHandlers, RealtimeModel, RealtimeSessionConfig } from "./types";
import { deepgramSttFactory, SttFactory, SttStream } from "./deepgram/stt";
import { deepgramTts, Tts, TtsSession } from "./deepgram/tts";

/**
 * A barge-in phone brain assembled from cheap parts instead of one pricey
 * speech-to-speech model: Deepgram streaming STT → text LLM (the shared
 * `provider`, i.e. Claude on Vertex) → Deepgram Aura TTS. It implements the same
 * RealtimeModel interface as the OpenAI Realtime client, so the Twilio bridge,
 * barge-in handling, tool calls, and web pairing are all unchanged — only the
 * per-minute cost drops (no audio-token pricing).
 *
 * Barge-in: Deepgram's VAD fires SpeechStarted → the bridge calls
 * cancelResponse(), which invalidates the in-flight turn so the LLM loop and the
 * TTS frame pump bail out immediately; the bridge also flushes Twilio playback.
 *
 * Every network dependency (STT, TTS, LLM) is injectable so the orchestration is
 * verified offline with fakes — see deepgramCascade.verify.ts.
 */

const FRAME_BYTES = 160; // 20 ms of G.711 μ-law @ 8 kHz (8 bytes/ms)
const VOICE_MAX_TOKENS = 120; // keep spoken replies short, snappy, and cheap

/**
 * Spanish tells for picking the TTS voice of a reply chunk. The STT tells us
 * what language the CALLER spoke; this catches the reply itself (the model can
 * legitimately answer "sí, claro que hablo español" to an English question).
 * Inverted punctuation, accented vowels, and the most common Spanish function
 * words — none of which appear in normal English prose.
 */
const ES_HINT =
  /[¿¡áéíóúñü]|\b(?:el|la|los|las|un|una|que|de|del|en|es|para|con|por|su|sus|como|hola|gracias|hablar|hablo|habla|puedo|puedes|ayudar|claro|bien|muy|ahora|tiene|tienes|hacer|llamada|reservar|cita|nombre|correo|trabajo|proyectos|dime|perdona|momento|entonces|porque|cuando|donde)\b/gi;

/**
 * Normalize model text into clean speakable prose before TTS. The persona asks
 * for plain sentences, but if any markdown or symbols slip through, reading
 * them aloud ("asterisk asterisk") is what makes a voice sound robotic —
 * scrub them here as a safety net.
 */
export function speakable(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ") // code blocks don't belong in speech
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1") // markdown links -> just the label
    .replace(/[*_`#>|]+/g, "") // emphasis / heading / table markers
    .replace(/^\s*[-•]\s+/gm, "") // bullet markers at line starts
    .replace(/[—–]/g, ", ") // dashes -> a natural spoken pause
    .replace(/&/g, " and ")
    .replace(/\s+/g, " ")
    .trim();
}

export interface CascadeDeps {
  sttFactory?: SttFactory;
  tts?: Tts;
  llm?: LlmProvider;
}

export class DeepgramCascade implements RealtimeModel {
  currentItemId: string | null = null;

  private readonly stt: SttStream;
  private readonly tts: Tts;
  /** One warm TTS socket per language voice ("en", "es"), created lazily. */
  private readonly ttsSessions = new Map<string, TtsSession>();
  private readonly llm: LlmProvider;
  /** Language the caller is currently speaking (from the STT's per-word tags). */
  private lang = "en";

  /** Wall-clock start of the current turn, for the [timing] diagnostics. */
  private turnT0 = 0;
  private firstAudioLogged = false;

  private readonly system: string;
  /** Per-call context; byte-stable for the whole call (built at call start). */
  private systemDynamic = "";
  private readonly tools: ToolDef[];
  private history: NeutralMessage[] = [];

  /** Bumped whenever a turn starts or is cancelled; stale turns bail on mismatch. */
  private turnSeq = 0;
  /** Resolves once the greeting's audio is fully synthesized; turns wait on it. */
  private greetingQueued: Promise<void> = Promise.resolve();
  /** Until this wall-clock time, echo matching is loosened (greeting playback). */
  private greetingLooseUntil = 0;
  private draining = false;
  private queue: string[] = [];
  private readonly pendingTools = new Map<string, (output: string) => void>();

  constructor(cfg: RealtimeSessionConfig, private h: RealtimeHandlers, deps: CascadeDeps = {}) {
    this.tts = deps.tts ?? deepgramTts;
    // Open the English voice's socket up front (one socket per voice for the
    // whole call — no per-utterance handshake); the Spanish voice connects the
    // first time it's needed. Tts impls without sessions (fakes) use stream().
    this.sessionFor("en");
    this.llm = deps.llm ?? defaultLlm;
    this.system = cfg.instructions;
    this.systemDynamic = cfg.instructionsDynamic ?? "";
    this.tools = cfg.tools.map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.parameters,
    }));

    const sttFactory = deps.sttFactory ?? deepgramSttFactory;
    this.stt = sttFactory.open({
      onSpeechStarted: () => this.h.onSpeechStarted?.(),
      onUtterance: (text, language) => this.onUtterance(text, language),
      isBotEcho: (text) => this.isBotEcho(text),
      onError: (e) => this.h.onError?.(e),
    });
    this.h.onReady?.();
  }

  // --- RealtimeModel surface -------------------------------------------------

  appendAudio(base64ulaw: string): void {
    this.stt.send(Buffer.from(base64ulaw, "base64"));
  }

  /**
   * A message typed on the paired website. Same pipeline as a spoken utterance
   * (queue + drain), so Elle answers it OUT LOUD on the call and the reply's
   * transcript streams back to the browser. The framing note tells the model
   * where the words came from without changing what the transcript shows.
   */
  sendText(text: string): void {
    const clean = text.trim();
    if (!clean) return;
    console.log(`[voice] caller (typed): "${clean}"`);
    this.h.onUserTranscript?.(clean);
    this.queue.push(`(Typed on the paired website, not spoken — answer out loud as usual.) ${clean}`);
    void this.drain();
  }

  cancelResponse(): void {
    // Invalidate the running turn. Its captured seq no longer matches turnSeq,
    // so the LLM loop and TTS pump stop at their next checkpoint. Twilio's own
    // playback buffer is flushed by the bridge (the `clear` message).
    this.turnSeq++;
  }

  /**
   * Speak an opening line the instant the call connects, before the caller says
   * anything — no LLM or STT round-trip, so there's no dead air at the start.
   * Deliberately kept OUT of the message history: the Messages API requires the
   * history to begin with a user turn, so the bridge instead notes in the
   * instructions that this line was already said, so the model won't repeat it.
   * The greeting is UNINTERRUPTIBLE: the bridge blocks barge-in until its audio
   * has fully played, and drain() waits on greetingQueued so no turn can bump
   * turnSeq (which would halt this speak() mid-synthesis).
   */
  greet(text: string): void {
    const clean = speakable(text);
    if (!clean) return;
    // Bilingual deploys close the greeting with one Spanish sentence, spoken
    // by the SPANISH voice (the English voice would mangle its pronunciation),
    // so Spanish-only callers hear in their own language that they can switch.
    const invite = spanishEnabled() ? SPANISH_INVITE : "";
    const full = invite ? `${clean} ${invite}` : clean;
    const seq = ++this.turnSeq;
    this.turnT0 = Date.now(); // keeps the greeting's timing log sane
    this.firstAudioLogged = false;
    this.currentItemId = `item_${seq}`;
    // Loose echo matching stays on while the greeting could still be playing;
    // ~80 ms/char deliberately over-estimates playback so the window outlasts it.
    this.greetingLooseUntil = Date.now() + Math.max(10_000, full.length * 80);
    this.h.onAssistantTranscript?.(full);
    this.greetingQueued = this.speak(clean, seq, "en")
      .then(() => (invite ? this.speak(invite, seq, "es") : undefined))
      .then(() => this.h.onGreetingSpoken?.())
      .catch((e) => this.h.onError?.(e));
  }

  truncate(_itemId: string, _audioEndMs: number): void {
    // No server-side audio buffer to truncate here — the cascade's "memory" is
    // the text history. Barge-in correctness comes from cancelResponse() halting
    // generation plus the bridge flushing Twilio. Kept for interface parity.
  }

  sendFunctionResult(callId: string, output: string): void {
    const resolve = this.pendingTools.get(callId);
    if (resolve) {
      this.pendingTools.delete(callId);
      resolve(output);
    }
  }

  close(): void {
    try {
      this.stt.close();
    } catch {
      /* already closed */
    }
    for (const session of this.ttsSessions.values()) {
      try {
        session.close();
      } catch {
        /* already closed */
      }
    }
    this.h.onClose?.();
  }

  // --- turn pipeline ---------------------------------------------------------

  private onUtterance(text: string, language?: string): void {
    const clean = text.trim();
    if (!clean) return;
    // Follow the caller's language (code-switching STT tags each utterance).
    // The reply voice and the error apology track this; the LLM mirrors on its
    // own from the transcript plus the persona's language rule.
    if (language && language !== this.lang && config.languages.supported.includes(language)) {
      console.log(`[voice] caller language: ${this.lang} -> ${language}`);
      this.lang = language;
    }
    console.log(`[voice] caller: "${clean}"`);
    this.h.onUserTranscript?.(clean);
    this.queue.push(clean);
    void this.drain();
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      // Never start a turn while the greeting is still being synthesized:
      // starting one bumps turnSeq, which would halt the greeting's TTS
      // mid-stream and cut it off. Once its audio is queued it's safe — the
      // bridge keeps playing it and blocks barge-in until it has fully aired.
      await this.greetingQueued;
      while (this.queue.length) {
        // Merge everything queued into ONE turn. A caller talking in bursts
        // (or over the greeting) otherwise triggers a separate LLM call and a
        // separate spoken answer per utterance — back-to-back replies to what
        // was really one thought. One merged turn is one API call and one
        // coherent answer.
        const parts = this.queue.splice(0);
        if (parts.length > 1) {
          console.log(`[voice] merged ${parts.length} queued utterances into one turn`);
        }
        try {
          await this.runTurn(parts.join(" "));
        } catch (err) {
          // A single failed turn (LLM/TTS error) must not crash the process and
          // drop the whole call. Say so instead of leaving dead air — silence
          // makes callers ask "are you there?".
          console.error("voice turn failed:", err);
          const seq = ++this.turnSeq;
          this.currentItemId = `item_${seq}`;
          try {
            const sorry =
              this.lang === "es"
                ? "Perdona, se me cruzó algo. ¿Me lo repites?"
                : "Sorry, I hit a snag just then. Could you say that again?";
            await this.speak(sorry, seq);
          } catch {
            /* if even the apology fails, stay alive for the next turn */
          }
        }
      }
    } finally {
      this.draining = false;
    }
  }

  /**
   * One LLM call for the turn. When the provider supports streaming, Elle
   * starts speaking as soon as the FIRST sentence of the reply exists, while
   * the rest is still being generated — the single biggest time-to-first-word
   * lever in the cascade. Returns the final response plus whether the text was
   * already spoken (non-streaming providers fall back to speak-after).
   */
  private async chatSpeaking(seq: number): Promise<{ resp: ChatResponse; spoke: boolean }> {
    const req = {
      systemStable: this.system,
      systemDynamic: this.systemDynamic,
      messages: this.history,
      tools: this.tools,
      maxTokens: VOICE_MAX_TOKENS,
      // The dynamic block is frozen at call start, so the conversation
      // breakpoint stays valid across the whole call.
      cacheConversation: true,
    };
    if (!this.llm.chatStream) {
      return { resp: await this.llm.chat(req), spoke: false };
    }

    let buffer = "";
    let chain: Promise<void> = Promise.resolve();
    let spoke = false;
    let sawFirstToken = false;
    const speakChunk = (chunk: string): void => {
      const clean = speakable(chunk);
      if (!clean || seq !== this.turnSeq) return;
      if (!spoke) console.log(`[timing] first sentence ready +${Date.now() - this.turnT0}ms`);
      spoke = true;
      this.currentItemId = `item_${seq}`;
      console.log(`[voice] elle (stream): "${clean}"`);
      this.h.onAssistantTranscript?.(clean);
      chain = chain.then(() => (seq === this.turnSeq ? this.speak(clean, seq) : undefined));
    };
    // Emit a chunk once a sentence has clearly ended (punctuation followed by
    // whitespace) and it's long enough to sound natural on its own.
    const SENTENCE_RE = /^([\s\S]{15,}?[.!?…]["')\]]?)\s+([\s\S]*)$/;
    const resp = await this.llm.chatStream(req, (delta) => {
      if (seq !== this.turnSeq) return;
      if (!sawFirstToken) {
        sawFirstToken = true;
        console.log(`[timing] llm first token +${Date.now() - this.turnT0}ms`);
      }
      buffer += delta;
      let m: RegExpExecArray | null;
      while ((m = SENTENCE_RE.exec(buffer))) {
        speakChunk(m[1]);
        buffer = m[2];
      }
    });
    if (buffer.trim() && seq === this.turnSeq) speakChunk(buffer);
    await chain; // let the audio finish queueing before tools / next hop run
    return { resp, spoke };
  }

  /**
   * chatSpeaking with one quick retry on transient API failures (overloaded /
   * 5xx / dropped stream). Anthropic's occasional "overloaded_error" mid-call
   * otherwise costs the whole turn and the caller hears an apology for what a
   * 400ms retry would have absorbed.
   */
  private async chatSpeakingWithRetry(seq: number): Promise<{ resp: ChatResponse; spoke: boolean }> {
    try {
      return await this.chatSpeaking(seq);
    } catch (err) {
      if (seq !== this.turnSeq) throw err;
      console.log(`[voice] llm call failed, retrying once: ${(err as Error)?.message?.slice(0, 80)}`);
      await new Promise((r) => setTimeout(r, 400));
      if (seq !== this.turnSeq) throw err;
      return await this.chatSpeaking(seq);
    }
  }

  private async runTurn(userText: string): Promise<void> {
    const seq = ++this.turnSeq;
    this.turnT0 = Date.now();
    this.firstAudioLogged = false;
    // Pre-warm the TTS socket while the LLM thinks: connect cost overlaps
    // generation instead of delaying the first audible word. Warm the voice for
    // the language the caller is speaking — the likely voice of the reply.
    this.sessionFor(this.lang)?.warm();
    this.history.push({ role: "user", text: userText });

    let { resp, spoke } = await this.chatSpeakingWithRetry(seq);
    if (seq !== this.turnSeq) return; // barged-in during the LLM call
    // Whether ANY audio went out this turn — segments before a tool call count.
    // The empty-reply nudge below must only fire when the caller truly heard
    // nothing, or the model repeats what it already said.
    let spokeAny = spoke;

    // Tool loop: surface each call to the bridge, wait for its result, continue.
    while (resp.toolCalls.length > 0) {
      this.history.push({
        role: "assistant",
        text: resp.text,
        toolCalls: resp.toolCalls,
        raw: resp.raw,
      });
      const results: ToolResultMsg[] = [];
      for (const call of resp.toolCalls) {
        const output = await this.awaitToolResult(call.name, call.input, call.id);
        results.push({ id: call.id, content: output, isError: false });
      }
      this.history.push({ role: "tool_results", results });
      if (seq !== this.turnSeq) return;

      ({ resp, spoke } = await this.chatSpeakingWithRetry(seq));
      if (seq !== this.turnSeq) return;
      spokeAny ||= spoke;
    }

    this.history.push({ role: "assistant", text: resp.text, toolCalls: [], raw: resp.raw });
    if (spoke) return; // streamed path already spoke everything
    let spoken = speakable(resp.text);
    if (!spoken && spokeAny) return; // turn was already voiced before the tool call — silence here is correct
    if (!spoken && seq === this.turnSeq) {
      // The model sometimes ends a tool turn (e.g. save_user_fact) with no text
      // at all — on a phone that's dead air and the caller says "hello?".
      // Nudge once for an actual spoken reply.
      console.log(`[voice] empty reply after turn — nudging for speech`);
      this.history.push({
        role: "user",
        text:
          "(The caller heard nothing — reply out loud now, one or two short sentences. " +
          "Do not mention saving notes, tools, or anything internal; just continue the conversation.)",
      });
      ({ resp, spoke } = await this.chatSpeakingWithRetry(seq));
      if (seq !== this.turnSeq) return;
      this.history.push({ role: "assistant", text: resp.text, toolCalls: [], raw: resp.raw });
      if (spoke) return;
      spoken = speakable(resp.text);
    }
    if (seq !== this.turnSeq) {
      console.log(`[voice] turn ${seq} canceled before speaking (barge-in)`);
      return;
    }
    console.log(`[voice] elle: "${spoken || "(empty)"}"`);
    if (!spoken) return;
    this.currentItemId = `item_${seq}`;
    this.h.onAssistantTranscript?.(spoken);
    await this.speak(spoken, seq);
  }

  private awaitToolResult(
    name: string,
    input: Record<string, unknown>,
    callId: string
  ): Promise<string> {
    return new Promise<string>((resolve) => {
      this.pendingTools.set(callId, resolve);
      this.h.onFunctionCall?.(name, input, callId);
    });
  }

  // --- echo suppression ------------------------------------------------------
  //
  // On a phone line, Elle's own voice can leak back through the caller's
  // handset and transcribe as "caller speech" — which both trips barge-in
  // (cutting her off mid-greeting) and lands as a bogus utterance she then
  // answers. Everything she says passes through speak(), so we keep a short
  // rolling window of her recent words; a transcript that is a verbatim chunk
  // of them is echo, not the caller.

  private spokenRecently: Array<{ text: string; at: number }> = [];
  private static readonly ECHO_WINDOW_MS = 30_000;

  private static normalizeForEcho(text: string): string {
    // \p{L}/\p{N} (not [a-z0-9]) so accented Spanish words survive intact —
    // otherwise "dímelo" becomes "d melo" and echo matching breaks in Spanish.
    return text
      .toLowerCase()
      .replace(/['’]/g, "")
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .trim();
  }

  private noteSpoken(text: string): void {
    const norm = DeepgramCascade.normalizeForEcho(text);
    if (!norm) return;
    const cutoff = Date.now() - DeepgramCascade.ECHO_WINDOW_MS;
    this.spokenRecently = this.spokenRecently.filter((s) => s.at >= cutoff);
    this.spokenRecently.push({ text: norm, at: Date.now() });
  }

  private isBotEcho(candidate: string): boolean {
    const norm = DeepgramCascade.normalizeForEcho(candidate);
    // Too short to attribute — leave it alone (it can't barge in anyway, and
    // dropping a real "yes"/"okay" would be worse than a stray echo word).
    if (norm.length < 8 || !norm.includes(" ")) return false;
    const cutoff = Date.now() - DeepgramCascade.ECHO_WINDOW_MS;
    const haystack = this.spokenRecently
      .filter((s) => s.at >= cutoff)
      .map((s) => s.text)
      .join(" ");
    if (haystack.includes(norm)) return true;
    // While the greeting could still be playing, also match loosely: line echo
    // arrives garbled (dropped words, misheard names), so an exact-substring
    // test misses it. A transcript made mostly of Elle's own recent words is
    // echo. Greeting-only on purpose — mid-call, callers legitimately repeat
    // her words back ("Tuesday at 2"), and loose matching would eat real turns.
    if (Date.now() < this.greetingLooseUntil) {
      const words = norm.split(" ");
      if (words.length >= 3) {
        const spoken = new Set(haystack.split(" "));
        const hits = words.filter((w) => spoken.has(w)).length;
        if (hits / words.length >= 0.75) return true;
      }
    }
    return false;
  }

  // --- language / voice selection --------------------------------------------

  private ttsModelFor(lang: string): string {
    return lang === "es" ? config.deepgram.ttsModelEs : config.deepgram.ttsModel;
  }

  private sessionFor(lang: string): TtsSession | null {
    if (!this.tts.openSession) return null;
    let session = this.ttsSessions.get(lang);
    if (!session) {
      session = this.tts.openSession(this.ttsModelFor(lang));
      this.ttsSessions.set(lang, session);
    }
    return session;
  }

  /**
   * Which voice should speak this reply chunk. The caller's language (STT) is
   * the default; a text scan overrides it when the reply itself is clearly the
   * other language — e.g. answering "do you speak Spanish?" in Spanish, or
   * quoting an English project name inside a Spanish call. Ambiguous chunks
   * stick with the caller's language.
   */
  private replyLang(text: string): string {
    if (!spanishEnabled()) return "en";
    const words = text.split(/\s+/).filter(Boolean).length;
    const hits = (text.match(ES_HINT) ?? []).length;
    if (hits >= 2 && hits >= words * 0.2) return "es";
    if (hits === 0 && words >= 4) return "en";
    return this.lang;
  }

  private async speak(text: string, seq: number, langOverride?: string): Promise<void> {
    this.noteSpoken(text);
    const lang = langOverride ?? this.replyLang(text);
    const itemId = this.currentItemId ?? `item_${seq}`;
    // Deepgram streams audio chunks as it synthesizes; forward them to Twilio as
    // soon as they arrive (Twilio buffers and plays at real time). Chunks aren't
    // frame-aligned, so carry the remainder and emit exact 160-byte μ-law frames.
    // Barge-in: cancelResponse() bumps turnSeq, so both the emit guard here and
    // the isCancelled() poll passed into the stream stop playback promptly.
    let carry: Buffer = Buffer.alloc(0);
    const emit = (ulaw: Buffer): void => {
      if (seq !== this.turnSeq) return;
      if (!this.firstAudioLogged) {
        this.firstAudioLogged = true;
        console.log(`[timing] first audio to caller +${Date.now() - this.turnT0}ms`);
      }
      carry = carry.length ? Buffer.concat([carry, ulaw]) : ulaw;
      while (carry.length >= FRAME_BYTES) {
        if (seq !== this.turnSeq) return;
        this.h.onAudioDelta?.(carry.subarray(0, FRAME_BYTES).toString("base64"), itemId);
        carry = carry.subarray(FRAME_BYTES);
      }
    };

    try {
      const session = this.sessionFor(lang);
      if (session) {
        await session.speak(text, emit, () => seq !== this.turnSeq);
      } else {
        await this.tts.stream(text, emit, () => seq !== this.turnSeq, this.ttsModelFor(lang));
      }
    } catch (e) {
      this.h.onError?.(e);
      return;
    }
    // Flush a final sub-frame remainder if the turn wasn't interrupted.
    if (seq === this.turnSeq && carry.length > 0) {
      this.h.onAudioDelta?.(carry.toString("base64"), itemId);
    }
  }
}

export const deepgramCascadeFactory: RealtimeFactory = (cfg, handlers) =>
  new DeepgramCascade(cfg, handlers);
