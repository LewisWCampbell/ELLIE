/**
 * Live smoke test for the Gemini Live path. Opens a real session against Vertex
 * AI, sends one text turn, collects the audio, and writes a playable WAV.
 *
 * The point is to separate three failures that all sound identical from a phone
 * call (silence): credentials that cannot reach Vertex, a model id that is not
 * enabled for this project or region, and an adapter bug. This exercises the
 * first two and nothing else, so if it passes and a call still fails, the
 * problem is on our side of the SDK.
 *
 * Run via tools/smoke-gemini-live.sh so the environment is loaded consistently.
 */
import fs from "node:fs";
import path from "node:path";
import { config } from "../src/config";

const OUT_DIR = "voice-previews";
const OUT_FILE = path.join(OUT_DIR, "gemini-live-smoke.wav");

/** The model emits PCM16 at this rate; the service resamples it to 8 kHz. */
const MODEL_RATE = 24000;

/** Wrap raw little-endian PCM16 in a minimal mono WAV header so it plays. */
function toWav(pcm: Buffer, sampleRate: number): Buffer {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16); // PCM fmt chunk size
  header.writeUInt16LE(1, 20); // format: PCM
  header.writeUInt16LE(1, 22); // channels: mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28); // byte rate
  header.writeUInt16LE(2, 32); // block align
  header.writeUInt16LE(16, 34); // bits per sample
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

async function main(): Promise<void> {
  const model = process.argv[2] || config.geminiLive.model;
  const project = config.vertex.projectId;
  const location = config.geminiLive.location;

  if (!project) {
    console.error("VERTEX_PROJECT_ID is not set. Nothing to authenticate against.");
    process.exit(2);
  }

  console.log(`project  ${project}`);
  console.log(`location ${location}`);
  console.log(`model    ${model}`);
  console.log(`voice    ${config.geminiLive.voice}`);
  console.log("connecting...");

  const { GoogleGenAI, Modality } = await import("@google/genai");
  const ai = new GoogleGenAI({ vertexai: true, project, location });

  const chunks: Buffer[] = [];
  let transcript = "";
  let failure: unknown = null;

  const done = new Promise<void>((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      resolve();
    };

    void (async () => {
      try {
        const session = await ai.live.connect({
          model,
          config: {
            responseModalities: [Modality.AUDIO],
            systemInstruction: "You are Elle, a warm phone assistant. Keep it to one sentence.",
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: config.geminiLive.voice } },
            },
            outputAudioTranscription: {},
          },
          callbacks: {
            onopen: () => console.log("connected"),
            onmessage: (m) => {
              const sc = m.serverContent;
              for (const part of sc?.modelTurn?.parts ?? []) {
                const data = part.inlineData?.data;
                if (data) chunks.push(Buffer.from(data, "base64"));
              }
              if (sc?.outputTranscription?.text) transcript += sc.outputTranscription.text;
              if (sc?.turnComplete) {
                session.close();
                finish();
              }
            },
            onerror: (e) => {
              failure = e;
              finish();
            },
            onclose: () => finish(),
          },
        });

        session.sendClientContent({
          turns: "Say hello and tell me you are working, in one short sentence.",
          turnComplete: true,
        });
      } catch (err) {
        failure = err;
        finish();
      }
    })();

    // Do not hang forever if the service accepts the socket and says nothing.
    setTimeout(() => {
      failure ??= new Error("timed out after 30s with no completed turn");
      finish();
    }, 30_000).unref();
  });

  await done;

  if (failure) {
    console.error("\nFAILED:", failure instanceof Error ? failure.message : failure);
    console.error(
      "\nMost likely causes, in order:\n" +
        "  1. This model id is not enabled for this project or region. Try another\n" +
        "     id, or another GEMINI_LIVE_LOCATION.\n" +
        "  2. Application Default Credentials are missing or lack Vertex access\n" +
        "     (`gcloud auth application-default login`, or the service account\n" +
        "     needs roles/aiplatform.user).\n" +
        "  3. The Vertex AI API is not enabled on the project."
    );
    process.exit(1);
  }

  const pcm = Buffer.concat(chunks);
  if (pcm.length === 0) {
    console.error("\nFAILED: the session opened but returned no audio.");
    process.exit(1);
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, toWav(pcm, MODEL_RATE));

  const seconds = pcm.length / 2 / MODEL_RATE;
  console.log(`\nOK: ${seconds.toFixed(1)}s of audio -> ${OUT_FILE}`);
  if (transcript) console.log(`she said: "${transcript.trim()}"`);
  console.log("\nThe voice architecture is reachable. Switch a call onto it with:");
  console.log("  'voice mode gcp for 10 minutes' in the intake channel");
}

void main();
