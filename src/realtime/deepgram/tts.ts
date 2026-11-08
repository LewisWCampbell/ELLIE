import WebSocket from "ws";
import { config } from "../../config";

/**
 * Streaming text-to-speech via Deepgram Aura's WebSocket. We ask for G.711
 * μ-law @ 8 kHz directly, so audio comes back in exactly the format Twilio wants
 * (no transcoding). Streaming (vs. a single REST call) means the first audio
 * arrives while the rest of the sentence is still being synthesized — the caller
 * hears Elle start almost immediately instead of after the whole reply is ready.
 *
 * The cascade depends only on the small interface below, so it's faked offline
 * (see deepgramCascade.verify).
 */

export interface Tts {
  /**
   * Synthesize `text`, invoking `onAudio` for each μ-law chunk as it arrives.
   * Resolves when synthesis finishes. `isCancelled` is polled so a barge-in
   * stops synthesis (and tears down the socket) promptly. `model` overrides the
   * configured voice (used for the per-language voices in bilingual mode).
   */
  stream(
    text: string,
    onAudio: (ulaw: Buffer) => void,
    isCancelled: () => boolean,
    model?: string
  ): Promise<void>;
  /**
   * Open a per-call session that keeps ONE WebSocket alive across every
   * utterance, so no reply (or sentence) pays the ~150-300 ms TLS+upgrade
   * handshake. Optional: callers fall back to stream() when absent. A voice is
   * fixed per socket, so bilingual calls hold one session per language.
   */
  openSession?(model?: string): TtsSession;
}

export interface TtsSession {
  /** Pre-open the socket (fire during LLM thinking so connect overlaps it). */
  warm(): void;
  /** Speak one chunk; resolves when its audio has fully arrived. */
  speak(text: string, onAudio: (ulaw: Buffer) => void, isCancelled: () => boolean): Promise<void>;
  close(): void;
}

class DeepgramTtsSession implements TtsSession {
  private ws: WebSocket | null = null;
  private opening: Promise<WebSocket> | null = null;
  private current: { onAudio: (b: Buffer) => void; isCancelled: () => boolean; done: () => void } | null = null;

  constructor(private model: string = config.deepgram.ttsModel) {}

  private connect(): Promise<WebSocket> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return Promise.resolve(this.ws);
    if (this.opening) return this.opening;
    const params = new URLSearchParams({
      model: this.model,
      encoding: "mulaw",
      sample_rate: "8000",
    });
    const ws = new WebSocket(`wss://api.deepgram.com/v1/speak?${params.toString()}`, {
      headers: { Authorization: `Token ${config.deepgram.apiKey}` },
    });
    this.opening = new Promise<WebSocket>((resolve, reject) => {
      ws.on("open", () => {
        this.ws = ws;
        this.opening = null;
        resolve(ws);
      });
      ws.on("error", (e) => {
        this.opening = null;
        if (this.ws === ws) this.ws = null;
        this.current?.done();
        reject(e);
      });
      ws.on("close", () => {
        if (this.ws === ws) this.ws = null;
        // If a speak was mid-flight, let it finish rather than hang.
        this.current?.done();
      });
      ws.on("message", (data: WebSocket.RawData, isBinary: boolean) => {
        const cur = this.current;
        if (!cur) return;
        if (cur.isCancelled()) {
          try {
            ws.send(JSON.stringify({ type: "Clear" }));
          } catch {
            /* socket gone */
          }
          cur.done();
          return;
        }
        if (isBinary) {
          cur.onAudio(data as Buffer);
          return;
        }
        try {
          const ev = JSON.parse(data.toString()) as { type?: string };
          if (ev.type === "Flushed") cur.done();
        } catch {
          /* ignore non-JSON */
        }
      });
    });
    return this.opening;
  }

  warm(): void {
    this.connect().catch(() => {
      /* pre-warm is best-effort; speak() retries the connect */
    });
  }

  async speak(text: string, onAudio: (ulaw: Buffer) => void, isCancelled: () => boolean): Promise<void> {
    if (isCancelled()) return;
    const ws = await this.connect();
    await new Promise<void>((resolve) => {
      let settled = false;
      this.current = {
        onAudio,
        isCancelled,
        done: () => {
          if (settled) return;
          settled = true;
          this.current = null;
          resolve();
        },
      };
      ws.send(JSON.stringify({ type: "Speak", text }));
      ws.send(JSON.stringify({ type: "Flush" }));
    });
  }

  close(): void {
    try {
      this.current?.done();
      this.ws?.close();
    } catch {
      /* already closed */
    }
    this.ws = null;
  }
}

export const deepgramTts: Tts = {
  openSession: (model?: string) => new DeepgramTtsSession(model),
  stream(text, onAudio, isCancelled, model): Promise<void> {
    const params = new URLSearchParams({
      model: model ?? config.deepgram.ttsModel,
      encoding: "mulaw",
      sample_rate: "8000",
    });
    const ws = new WebSocket(`wss://api.deepgram.com/v1/speak?${params.toString()}`, {
      headers: { Authorization: `Token ${config.deepgram.apiKey}` },
    });

    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (err?: unknown): void => {
        if (settled) return;
        settled = true;
        try {
          ws.close();
        } catch {
          /* already closed */
        }
        if (err) reject(err);
        else resolve();
      };

      ws.on("open", () => {
        ws.send(JSON.stringify({ type: "Speak", text }));
        ws.send(JSON.stringify({ type: "Flush" }));
      });

      ws.on("message", (data: WebSocket.RawData, isBinary: boolean) => {
        if (isCancelled()) {
          try {
            ws.send(JSON.stringify({ type: "Clear" }));
          } catch {
            /* socket gone */
          }
          finish();
          return;
        }
        if (isBinary) {
          onAudio(data as Buffer); // a chunk of μ-law audio
          return;
        }
        // Text control message: "Flushed" means all buffered text was spoken.
        try {
          const ev = JSON.parse(data.toString()) as { type?: string };
          if (ev.type === "Flushed") finish();
        } catch {
          /* ignore non-JSON */
        }
      });

      ws.on("error", (e) => finish(e));
      ws.on("close", () => finish());
    });
  },
};
