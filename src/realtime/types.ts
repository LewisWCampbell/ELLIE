/**
 * Provider-neutral contract for a realtime speech-to-speech model, so the
 * Twilio bridge depends on this interface rather than on OpenAI directly.
 * Removing the OpenAI dependency = providing a different RealtimeFactory
 * (e.g. a self-hosted Moshi client) that implements RealtimeModel.
 */

export interface RealtimeTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface RealtimeSessionConfig {
  /**
   * The static part of the system prompt: persona, inline knowledge base,
   * greeting note. BYTE-IDENTICAL across every call, so the API's prompt cache
   * shares one entry across all calls (the first call in a cache window writes
   * it; every other call reads it at ~10% price with faster prefill).
   */
  instructions: string;
  /** Per-call context (caller facts, pairing code, scheduling times). Rendered
   *  after the static block so it never breaks the shared cache prefix. */
  instructionsDynamic?: string;
  voice: string;
  tools: RealtimeTool[];
}

export interface RealtimeHandlers {
  onReady?: () => void;
  /** Base64 μ-law @ 8 kHz audio to play to the caller, plus the model item id. */
  onAudioDelta?: (base64ulaw: string, itemId: string) => void;
  /** Caller started talking over the bot (barge-in). */
  onSpeechStarted?: () => void;
  /**
   * The greet() opening line has finished generating: every frame of its audio
   * has been handed to onAudioDelta. The bridge uses this to keep barge-in
   * disabled until the greeting has fully PLAYED — the caller always hears the
   * whole opening message.
   */
  onGreetingSpoken?: () => void;
  onUserTranscript?: (text: string) => void;
  onAssistantTranscript?: (text: string) => void;
  onFunctionCall?: (name: string, args: Record<string, unknown>, callId: string) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
}

/** The methods the Twilio bridge calls on whatever realtime model is behind it. */
export interface RealtimeModel {
  /** Id of the model's current spoken item, for truncation on barge-in. */
  readonly currentItemId: string | null;
  appendAudio(base64ulaw: string): void;
  /**
   * Feed a TEXT message from the caller into the conversation (typed on the
   * paired website). Treated like a spoken utterance: it queues a turn and the
   * reply is spoken on the call. Optional — models without it fall back to the
   * pendingWebNotes path.
   */
  sendText?(text: string): void;
  cancelResponse(): void;
  truncate(itemId: string, audioEndMs: number): void;
  sendFunctionResult(callId: string, output: string): void;
  /**
   * Speak an opening line immediately, before the caller says anything, so the
   * call doesn't open with silence. Optional — a model that greets on its own
   * (or can't) simply doesn't implement it.
   */
  greet?(text: string): void;
  close(): void;
}

export type RealtimeFactory = (
  config: RealtimeSessionConfig,
  handlers: RealtimeHandlers
) => RealtimeModel;
