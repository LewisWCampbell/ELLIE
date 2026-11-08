import WebSocket from "ws";
import { config } from "../../config";

/**
 * Streaming speech-to-text over Deepgram's realtime WebSocket. Twilio Media
 * Streams speak G.711 μ-law @ 8 kHz, and Deepgram accepts exactly that
 * (`encoding=mulaw&sample_rate=8000`), so there's no transcoding on the way in.
 *
 * The stream does its own endpointing and voice-activity detection, so it can
 * tell the cascade two things: the caller *started* talking (barge-in), and the
 * caller *finished* an utterance (time to answer). The cascade depends on the
 * small interface below, so it can be faked offline (see deepgramCascade.verify).
 */

export interface SttHandlers {
  /** Caller started speaking — used to trigger barge-in. */
  onSpeechStarted?: () => void;
  /**
   * A complete caller utterance (assembled from final results). `language` is
   * the dominant language it was spoken in (e.g. "en", "es") when the stream
   * runs a code-switching model; undefined on single-language streams.
   */
  onUtterance?: (text: string, language?: string) => void;
  /**
   * Returns true when a transcript is an echo of the bot's own voice leaking
   * back through the caller's handset. Echoed text must neither barge in nor
   * count as caller speech — without this the long opening greeting gets cut
   * off the moment its own echo transcribes past the barge-in threshold.
   */
  isBotEcho?: (text: string) => boolean;
  onError?: (err: unknown) => void;
}

export interface SttStream {
  /** Feed one chunk of caller audio (raw μ-law bytes). */
  send(ulaw: Buffer): void;
  close(): void;
}

export interface SttFactory {
  open(handlers: SttHandlers): SttStream;
}

class DeepgramSttStream implements SttStream {
  private ws: WebSocket;
  private ready = false;
  private pending: Buffer[] = [];
  private finals: Array<{ text: string; langWords: Map<string, number> }> = [];
  /** Whether we've already flagged barge-in for the utterance in progress. */
  private barged = false;

  constructor(private h: SttHandlers) {
    // Trailing silence before we treat the caller's turn as finished. Deepgram
    // requires utterance_end_ms >= 1000, so keep the fallback boundary at least
    // as long as the endpointing window.
    const endpointing = config.deepgram.endpointingMs;
    const utteranceEndMs = String(Math.max(1000, parseInt(endpointing, 10) || 1000));
    // Bilingual deploys use the code-switching model with language=multi so a
    // caller can move between languages mid-call; each result's words carry the
    // language they were spoken in, which we surface per utterance.
    const multilingual = config.languages.supported.length > 1;
    const params = new URLSearchParams({
      encoding: "mulaw",
      sample_rate: "8000",
      channels: "1",
      model: multilingual ? config.deepgram.sttMultiModel : config.deepgram.sttModel,
      interim_results: "true",
      vad_events: "true", // emit SpeechStarted for barge-in
      endpointing, // ms of silence that ends a phrase (default 1000)
      utterance_end_ms: utteranceEndMs, // fallback utterance boundary
      smart_format: "true",
      punctuate: "true",
    });
    if (multilingual) params.set("language", "multi");
    this.ws = new WebSocket(`wss://api.deepgram.com/v1/listen?${params.toString()}`, {
      headers: { Authorization: `Token ${config.deepgram.apiKey}` },
    });
    this.ws.on("open", () => {
      this.ready = true;
      for (const b of this.pending) this.ws.send(b);
      this.pending = [];
    });
    this.ws.on("message", (d) => this.onMessage(d));
    this.ws.on("error", (e) => this.h.onError?.(e));
  }

  send(ulaw: Buffer): void {
    if (this.ready && this.ws.readyState === WebSocket.OPEN) this.ws.send(ulaw);
    else this.pending.push(ulaw);
  }

  private flushUtterance(source: string): void {
    const text = this.finals.map((f) => f.text).join(" ").replace(/\s+/g, " ").trim();
    // Dominant language across the utterance's words (code-switching streams
    // tag each word). Majority wins so one borrowed word can't flip the voice.
    const tally = new Map<string, number>();
    for (const f of this.finals) {
      for (const [lang, n] of f.langWords) tally.set(lang, (tally.get(lang) ?? 0) + n);
    }
    let language: string | undefined;
    let best = 0;
    for (const [lang, n] of tally) {
      if (n > best) [language, best] = [lang, n];
    }
    this.finals = [];
    this.barged = false; // next utterance gets a fresh barge-in decision
    if (text) {
      // speech_final = the fast endpointing path; UtteranceEnd = the >=1s
      // fallback. If turns keep flushing via the fallback, endpointing is
      // failing (noise floor keeping VAD open) and replies feel slow.
      console.log(`[timing] utterance end via ${source}`);
      this.h.onUtterance?.(text, language);
    }
  }

  private onMessage(data: WebSocket.RawData): void {
    let ev: Record<string, unknown> & { type?: string };
    try {
      ev = JSON.parse(data.toString());
    } catch {
      return;
    }
    // Note: we deliberately do NOT barge in on the raw `SpeechStarted` VAD event
    // — it fires on any sound (breath, cough, line noise, echo of Elle's own
    // voice) and was cutting her off. Barge-in is gated on recognized words
    // below instead.
    if (ev.type === "UtteranceEnd") {
      this.flushUtterance("UtteranceEnd (slow fallback)");
      return;
    }
    if (ev.type === "Results") {
      const channel = ev.channel as
        | {
            alternatives?: Array<{
              transcript?: string;
              languages?: string[];
              words?: Array<{ language?: string }>;
            }>;
          }
        | undefined;
      const alt = channel?.alternatives?.[0];
      const text = (alt?.transcript ?? "").trim();
      const echo = text.length > 0 && (this.h.isBotEcho?.(text) ?? false);
      if (echo) {
        if (ev.is_final) console.log(`[voice] echo of own voice ignored: "${text}"`);
      } else {
        // Treat the caller as interrupting only once enough *recognized speech*
        // has landed (interim or final). Filters out noise that trips raw VAD.
        if (!this.barged && text.length >= config.deepgram.bargeInMinChars) {
          this.barged = true;
          console.log(`[voice] barge-in trigger: "${text}"`);
          this.h.onSpeechStarted?.();
        }
        if (ev.is_final && text) {
          // Per-word language tags from language=multi; fall back to the
          // result-level languages array (first entry = dominant) if absent.
          const langWords = new Map<string, number>();
          for (const w of alt?.words ?? []) {
            if (w.language) langWords.set(w.language, (langWords.get(w.language) ?? 0) + 1);
          }
          if (langWords.size === 0 && alt?.languages?.[0]) langWords.set(alt.languages[0], 1);
          this.finals.push({ text, langWords });
        }
      }
      // speech_final marks the end of a spoken turn (silence after speech).
      if (ev.speech_final) this.flushUtterance("speech_final (fast path)");
    }
  }

  close(): void {
    try {
      if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "CloseStream" }));
      this.ws.close();
    } catch {
      /* already closed */
    }
  }
}

export const deepgramSttFactory: SttFactory = {
  open: (h) => new DeepgramSttStream(h),
};
