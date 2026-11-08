import WebSocket from "ws";
import { config } from "../config";
import { RealtimeFactory, RealtimeHandlers, RealtimeModel, RealtimeSessionConfig } from "./types";

/**
 * Thin client for the OpenAI Realtime API (speech-to-speech over WebSocket).
 * The model does STT + reasoning + TTS + turn detection itself; we only relay
 * audio and react to a few events. Barge-in is native — server VAD emits
 * `input_audio_buffer.speech_started` the moment the caller talks over the bot.
 *
 * This is one implementation of the RealtimeModel interface (see ./types).
 * To remove the OpenAI dependency, provide a different RealtimeFactory — the
 * Twilio bridge doesn't know or care which model is behind it.
 *
 * NOTE: the Realtime API has evolved since its 2024 preview. Field names here
 * follow the widely-used preview shape; if you're on a newer GA model
 * (`gpt-realtime`+), check the current reference and adjust the `session.update`
 * payload and event names in one place — this file.
 */

export class OpenAIRealtime implements RealtimeModel {
  private ws: WebSocket;
  responseActive = false;
  currentItemId: string | null = null;

  constructor(private sessionConfig: RealtimeSessionConfig, private h: RealtimeHandlers) {
    const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(config.realtime.model)}`;
    this.ws = new WebSocket(url, {
      headers: {
        Authorization: `Bearer ${config.realtime.apiKey}`,
        "OpenAI-Beta": "realtime=v1",
      },
    });
    this.ws.on("open", () => this.configure());
    this.ws.on("message", (data) => this.onMessage(data));
    this.ws.on("error", (e) => this.h.onError?.(e));
    this.ws.on("close", () => this.h.onClose?.());
  }

  private send(obj: unknown): void {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  }

  private configure(): void {
    this.send({
      type: "session.update",
      session: {
        modalities: ["audio", "text"],
        instructions:
          this.sessionConfig.instructions +
          (this.sessionConfig.instructionsDynamic
            ? `\n\n${this.sessionConfig.instructionsDynamic}`
            : ""),
        voice: this.sessionConfig.voice,
        // Twilio Media Streams speak G.711 μ-law @ 8 kHz; match it both ways so
        // the bridge is a passthrough with no transcoding.
        input_audio_format: "g711_ulaw",
        output_audio_format: "g711_ulaw",
        input_audio_transcription: { model: "whisper-1" },
        turn_detection: {
          type: "server_vad",
          threshold: 0.5,
          prefix_padding_ms: 300,
          silence_duration_ms: 500,
        },
        tools: this.sessionConfig.tools.map((t) => ({
          type: "function",
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
        tool_choice: "auto",
      },
    });
    this.h.onReady?.();
  }

  /** Forward one 20 ms frame of caller audio (base64 μ-law) to the model. */
  appendAudio(base64ulaw: string): void {
    this.send({ type: "input_audio_buffer.append", audio: base64ulaw });
  }

  /** Stop the in-progress spoken response (used on barge-in). */
  cancelResponse(): void {
    if (this.responseActive) this.send({ type: "response.cancel" });
  }

  /**
   * Tell the model the caller only heard `audioEndMs` of its last utterance, so
   * its memory of "what I said" matches reality. This is the "retain & adapt"
   * piece — without it the model thinks it finished a sentence the caller cut off.
   */
  truncate(itemId: string, audioEndMs: number): void {
    if (audioEndMs <= 0) return;
    this.send({
      type: "conversation.item.truncate",
      item_id: itemId,
      content_index: 0,
      audio_end_ms: audioEndMs,
    });
  }

  /** Return a tool result and let the model continue speaking. */
  sendFunctionResult(callId: string, output: string): void {
    this.send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: callId, output },
    });
    this.send({ type: "response.create" });
  }

  close(): void {
    try {
      this.ws.close();
    } catch {
      /* already closed */
    }
  }

  private onMessage(data: WebSocket.RawData): void {
    let ev: Record<string, unknown> & { type?: string };
    try {
      ev = JSON.parse(data.toString());
    } catch {
      return;
    }
    switch (ev.type) {
      case "response.created":
        this.responseActive = true;
        break;
      case "response.done":
        this.responseActive = false;
        break;
      case "response.output_item.added":
        if (ev.item && typeof (ev.item as { id?: string }).id === "string") {
          this.currentItemId = (ev.item as { id: string }).id;
        }
        break;
      case "response.audio.delta":
        if (typeof ev.delta === "string") {
          if (typeof ev.item_id === "string") this.currentItemId = ev.item_id;
          this.h.onAudioDelta?.(ev.delta, this.currentItemId ?? "");
        }
        break;
      case "input_audio_buffer.speech_started":
        this.h.onSpeechStarted?.();
        break;
      case "conversation.item.input_audio_transcription.completed":
        if (typeof ev.transcript === "string") this.h.onUserTranscript?.(ev.transcript);
        break;
      case "response.audio_transcript.done":
        if (typeof ev.transcript === "string") this.h.onAssistantTranscript?.(ev.transcript);
        break;
      case "response.function_call_arguments.done": {
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse((ev.arguments as string) || "{}");
        } catch {
          /* malformed args — pass empty */
        }
        this.h.onFunctionCall?.(ev.name as string, args, ev.call_id as string);
        break;
      }
      case "error":
        this.h.onError?.(ev.error ?? ev);
        break;
    }
  }
}

/** Factory the Twilio bridge uses by default. Swap this to drop OpenAI. */
export const openAIRealtimeFactory: RealtimeFactory = (config, handlers) =>
  new OpenAIRealtime(config, handlers);
