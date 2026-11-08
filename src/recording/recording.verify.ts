/**
 * Offline verification of call recording — no network, no GCS. Exercises the
 * μ-law → WAV encoder, the two-channel recorder with silence alignment, and the
 * BlobStore contract via a fake store.
 *
 * Run: npx tsx src/recording/recording.verify.ts
 */
import assert from "node:assert";
import { encodeStereoWav, ulawByteToPcm16, ULAW_SILENCE } from "./wav";
import { CallRecorder } from "./recorder";
import { BlobStore } from "./blob";

async function main(): Promise<void> {
// --- WAV header + interleaving -------------------------------------------
{
  const caller = Buffer.from([0x00, 0x10, 0x20, 0x30]); // 4 μ-law samples
  const bot = Buffer.from([0x40, 0x50]); // 2 samples → padded to 4 with silence
  const wav = encodeStereoWav(caller, bot);

  assert.strictEqual(wav.toString("ascii", 0, 4), "RIFF", "RIFF magic");
  assert.strictEqual(wav.toString("ascii", 8, 12), "WAVE", "WAVE magic");
  assert.strictEqual(wav.toString("ascii", 12, 16), "fmt ", "fmt chunk");
  assert.strictEqual(wav.readUInt16LE(20), 1, "PCM format");
  assert.strictEqual(wav.readUInt16LE(22), 2, "stereo");
  assert.strictEqual(wav.readUInt32LE(24), 8000, "8 kHz");
  assert.strictEqual(wav.readUInt16LE(34), 16, "16-bit samples");
  const dataLen = 4 /*samples*/ * 2 /*ch*/ * 2 /*bytes*/;
  assert.strictEqual(wav.readUInt32LE(40), dataLen, "data length");
  assert.strictEqual(wav.length, 44 + dataLen, "total file size");

  // Left channel of sample 0 = decoded caller[0]; right of last sample = silence.
  assert.strictEqual(wav.readInt16LE(44), ulawByteToPcm16(0x00), "left ch sample 0");
  assert.strictEqual(
    wav.readInt16LE(44 + 3 * 4 + 2),
    ulawByteToPcm16(ULAW_SILENCE),
    "right ch padded with silence"
  );
}

// --- recorder aligns the bot channel to the caller clock ------------------
{
  const rec = new CallRecorder();
  assert.ok(rec.isEmpty, "starts empty");
  // 320 bytes of caller audio arrive first (40 ms), then the bot speaks 160 bytes.
  rec.pushCaller(Buffer.alloc(320, 0x22));
  rec.pushBot(Buffer.alloc(160, 0x44));
  rec.pushCaller(Buffer.alloc(160, 0x22));
  assert.ok(!rec.isEmpty, "captured audio");
  // Bot channel was silence-padded to the caller's 320 before its 160 → 480 total;
  // caller is 320+160 = 480. Duration = 480 bytes / 8 = 60 ms.
  assert.strictEqual(rec.durationMs, 60, "duration from the longer channel");

  const wav = rec.toWav();
  const samples = 480;
  assert.strictEqual(wav.readUInt32LE(40), samples * 4, "both channels span the call");
  // First bot sample should be silence (padding), not the 0x44 tone.
  assert.strictEqual(wav.readInt16LE(44 + 2), ulawByteToPcm16(ULAW_SILENCE), "bot starts silent");
  // At 40 ms (sample 320) the bot tone begins.
  assert.strictEqual(wav.readInt16LE(44 + 320 * 4 + 2), ulawByteToPcm16(0x44), "bot tone aligned to ~40 ms");
}

// --- BlobStore contract via a fake ---------------------------------------
{
  const puts: Array<{ path: string; type: string; bytes: number }> = [];
  const fake: BlobStore = {
    async put(path, data, contentType) {
      puts.push({ path, type: contentType, bytes: data.length });
      return `mem://${path}`;
    },
  };
  const rec = new CallRecorder();
  rec.pushCaller(Buffer.alloc(160, 0x10));
  rec.pushBot(Buffer.alloc(160, 0x20));
  const wavUri = await fake.put("recordings/2026-08-17/CA1.wav", rec.toWav(), "audio/wav");
  await fake.put("recordings/2026-08-17/CA1.json", Buffer.from('{"callSid":"CA1"}'), "application/json");
  assert.strictEqual(wavUri, "mem://recordings/2026-08-17/CA1.wav", "put returns a uri");
  assert.strictEqual(puts.length, 2, "wav + json uploaded");
  assert.strictEqual(puts[0].type, "audio/wav", "wav content type");
  assert.strictEqual(puts[1].type, "application/json", "json content type");
  assert.ok(puts[0].bytes > 44, "wav has header + data");
}

  console.log("Recording verification passed:");
  console.log("  ✓ μ-law decodes and a valid 16-bit stereo WAV header is written");
  console.log("  ✓ recorder pads the bot channel with silence to align to the caller clock");
  console.log("  ✓ BlobStore receives the WAV and JSON sidecar with correct content types");
}

void main();
