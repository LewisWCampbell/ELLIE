import { config } from "../config";
import { CallerAudioUpsampler, ModelAudioDownsampler } from "./audio/g711";
import { RealtimeFactory, RealtimeHandlers, RealtimeModel, RealtimeSessionConfig } from "./types";

/**
 * Gemini Live on Vertex AI: a fused speech-to-speech model, the same shape of
 * architecture as the OpenAI Realtime path but running inside Google Cloud.
 *
 * Why this exists alongside openaiRealtime.ts:
 *  - No new credential. Auth is Application Default Credentials, so on Cloud
 *    Run it is the service account that is already there. Nothing to mint,
 *    store, rotate, or leak.
 *  - Same BAA. Vertex is covered by the GCP BAA that already covers the
 *    compute and Secret Manager, so no separate model-vendor agreement.
 *  - Same bill, same project, same audit trail as everything else here.
 *
 * The cost of that: unlike Twilio, Deepgram, and the OpenAI Realtime API, this
 * model does not speak G.711. Caller audio has to be decoded and upsampled to
 * PCM16 16 kHz on the way in, and the model's PCM16 24 kHz output resampled
 * back down to μ-law 8 kHz on the way out. That conversion lives in
 * ./audio/g711.ts and is verified offline; everything else here is protocol.
 *
 * This is one implementation of RealtimeModel (see ./types). The Twilio bridge
 * does not know which model is behind it.
 */

/** Gemini Live speaks PCM16 at these rates. Not negotiable, unlike the others. */
const INPUT_MIME = "audio/pcm;rate=16000";

/**
 * The slice of the SDK's session this adapter actually uses, declared
 * structurally rather than imported.
 *
 * Two reasons. The SDK is ESM-only and this service is CommonJS, so importing
 * its types from here needs a resolution-mode dance that buys nothing. More
 * usefully, depending on a shape rather than a class means the offline verify
 * can hand this adapter a fake session and exercise every path (audio in,
 * audio out, barge-in, tools, greeting) with no credentials and no network.
 */
export interface LiveSession {
  sendRealtimeInput(params: { audio: { data: string; mimeType: string } }): void;
  sendClientContent(params: { turns: string; turnComplete: boolean }): void;
  sendToolResponse(params: {
    functionResponses: Array<{ id: string; name: string; response: Record<string, unknown> }>;
  }): void;
  close(): void;
}

/** The slice of a Live server message this adapter reacts to. */
export interface LiveMessage {
  serverContent?: {
    modelTurn?: { parts?: Array<{ inlineData?: { data?: string } }> };
    interrupted?: boolean;
    turnComplete?: boolean;
    inputTranscription?: { text?: string };
    outputTranscription?: { text?: string };
  };
  toolCall?: {
    functionCalls?: Array<{ id?: string; name?: string; args?: Record<string, unknown> }>;
  };
}

export interface LiveCallbacks {
  onopen: () => void;
  onmessage: (m: LiveMessage) => void;
  onerror: (e: unknown) => void;
  onclose: () => void;
}

/** How a session is opened. Swapped for a fake in the offline verify. */
export type LiveConnector = (
  instructions: string,
  sessionConfig: RealtimeSessionConfig,
  callbacks: LiveCallbacks
) => Promise<LiveSession>;

/**
 * The real connector: Gemini Live on Vertex AI, authenticated by Application
 * Default Credentials. The SDK is loaded dynamically because it is ESM-only
 * and this service is CommonJS; doing it here also means a deploy that never
 * switches to this architecture never pays to load it.
 */
export const vertexLiveConnector: LiveConnector = async (instructions, sessionConfig, cb) => {
  const { GoogleGenAI, Modality } = await import("@google/genai");
  const ai = new GoogleGenAI({
    vertexai: true,
    project: config.vertex.projectId,
    location: config.geminiLive.location,
  });

  const session = await ai.live.connect({
    model: config.geminiLive.model,
    config: {
      // Audio only. Transcripts come from the transcription configs below
      // rather than as a second response modality, which the Live API does not
      // allow alongside audio.
      responseModalities: [Modality.AUDIO],
      systemInstruction: instructions,
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: config.geminiLive.voice } },
      },
      // Both sides transcribed so the call log, the paired browser, and the
      // admin transcript view work exactly as they do on the cascade. A fused
      // model that cannot be read back is a demo, not an operable system.
      inputAudioTranscription: {},
      outputAudioTranscription: {},
      tools: [
        {
          functionDeclarations: sessionConfig.tools.map((t) => ({
            name: t.name,
            description: t.description,
            parameters: t.parameters as never,
          })),
        },
      ],
    },
    callbacks: {
      onopen: cb.onopen,
      onmessage: (m) => cb.onmessage(m as LiveMessage),
      onerror: cb.onerror,
      onclose: cb.onclose,
    },
  });

  return session as unknown as LiveSession;
};

export class GeminiLive implements RealtimeModel {
  private session: LiveSession | null = null;
  private closed = false;
  /** Frames that arrived before the socket finished opening. */
  private queued: string[] = [];
  private up = new CallerAudioUpsampler();
  private down = new ModelAudioDownsampler();

  /**
   * Gemini Live has no per-item ids to truncate against, so a synthetic turn id
   * stands in. The bridge only uses it to correlate audio with a truncate call,
   * and truncation is a no-op here (see truncate()).
   */
  private turn = 0;
  currentItemId: string | null = null;

  /** True between the first audio of a greeting and its turn completing. */
  private greeting = false;

  constructor(
    private sessionConfig: RealtimeSessionConfig,
    private h: RealtimeHandlers,
    private connector: LiveConnector = vertexLiveConnector
  ) {
    void this.connect();
  }

  private async connect(): Promise<void> {
    const instructions =
      this.sessionConfig.instructions +
      (this.sessionConfig.instructionsDynamic
        ? `\n\n${this.sessionConfig.instructionsDynamic}`
        : "");

    try {
      this.session = await this.connector(instructions, this.sessionConfig, {
        onopen: () => {
          // Anything the caller said while the socket was opening still counts:
          // a caller who starts talking immediately is common.
          for (const frame of this.queued) this.pushAudio(frame);
          this.queued = [];
          this.h.onReady?.();
        },
        onmessage: (m) => this.onMessage(m),
        onerror: (e) => this.h.onError?.(e),
        onclose: () => this.h.onClose?.(),
      });

      // close() may have been called while the connect promise was in flight.
      if (this.closed) this.session.close();
    } catch (err) {
      this.h.onError?.(err);
    }
  }

  /** Forward one 20 ms frame of caller audio (base64 μ-law @ 8 kHz). */
  appendAudio(base64ulaw: string): void {
    if (!this.session) {
      // Bounded: ~2 seconds of frames. A socket that never opens should not
      // grow a buffer for the length of the call.
      if (this.queued.length < 100) this.queued.push(base64ulaw);
      return;
    }
    this.pushAudio(base64ulaw);
  }

  private pushAudio(base64ulaw: string): void {
    const pcm16k = this.up.convert(Buffer.from(base64ulaw, "base64"));
    try {
      this.session?.sendRealtimeInput({
        audio: { data: pcm16k.toString("base64"), mimeType: INPUT_MIME },
      });
    } catch (err) {
      this.h.onError?.(err);
    }
  }

  /** Typed input from a paired browser, spoken back on the call. */
  sendText(text: string): void {
    try {
      this.session?.sendClientContent({ turns: text, turnComplete: true });
    } catch (err) {
      this.h.onError?.(err);
    }
  }

  /**
   * Barge-in. The model runs its own duplex turn-taking and stops generating
   * on its own when the caller speaks, announcing it with `interrupted`. There
   * is no client-side cancel to send, so this only drops the resampler's
   * partial group: without that, the tail of the abandoned turn would be
   * prepended to the first frame of the next one.
   */
  cancelResponse(): void {
    this.down.flush();
  }

  /**
   * No-op. Truncation exists so a model that believes it finished a sentence
   * can be told how much the caller actually heard. Gemini Live tracks that
   * itself as part of handling the interruption, so there is nothing to
   * correct, and inventing a correction would fight the model's own state.
   */
  truncate(_itemId: string, _audioEndMs: number): void {
    /* handled model-side */
  }

  sendFunctionResult(callId: string, output: string): void {
    let response: Record<string, unknown>;
    try {
      response = JSON.parse(output);
    } catch {
      // Tools here return JSON, but a plain string should not break the turn.
      response = { result: output };
    }
    try {
      this.session?.sendToolResponse({
        functionResponses: [{ id: callId, name: callId, response }],
      });
    } catch (err) {
      this.h.onError?.(err);
    }
  }

  /** Speak an opening line before the caller has said anything. */
  greet(text: string): void {
    this.greeting = true;
    this.sendText(
      `Open the call by saying exactly this, and nothing else: "${text}". Then wait for the caller.`
    );
  }

  close(): void {
    this.closed = true;
    try {
      this.session?.close();
    } catch {
      /* already closed */
    }
  }

  private onMessage(m: LiveMessage): void {
    // Tool calls first: they can arrive alongside content.
    for (const call of m.toolCall?.functionCalls ?? []) {
      this.h.onFunctionCall?.(
        call.name ?? "",
        (call.args as Record<string, unknown>) ?? {},
        call.id ?? call.name ?? ""
      );
    }

    const sc = m.serverContent;
    if (!sc) return;

    if (sc.interrupted) {
      this.down.flush();
      this.h.onSpeechStarted?.();
    }

    if (sc.inputTranscription?.text) this.h.onUserTranscript?.(sc.inputTranscription.text);
    if (sc.outputTranscription?.text) this.h.onAssistantTranscript?.(sc.outputTranscription.text);

    for (const part of sc.modelTurn?.parts ?? []) {
      const data = part.inlineData?.data;
      if (!data) continue;
      this.currentItemId = `turn-${this.turn}`;
      const ulaw = this.down.convert(Buffer.from(data, "base64"));
      if (ulaw.length) this.h.onAudioDelta?.(ulaw.toString("base64"), this.currentItemId);
    }

    if (sc.turnComplete) {
      // Flush the resampler's remainder so the last few milliseconds of a turn
      // are not held back until the next one starts.
      const tail = this.down.flush();
      if (tail.length) this.h.onAudioDelta?.(tail.toString("base64"), this.currentItemId ?? "");
      this.turn++;
      if (this.greeting) {
        this.greeting = false;
        this.h.onGreetingSpoken?.();
      }
    }
  }
}

export const geminiLiveFactory: RealtimeFactory = (sessionConfig, handlers) =>
  new GeminiLive(sessionConfig, handlers);
