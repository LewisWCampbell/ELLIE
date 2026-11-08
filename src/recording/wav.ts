/**
 * Assemble a stereo WAV from the two G.711 μ-law 8 kHz streams of a call:
 * left channel = caller, right channel = Elle. We decode μ-law to 16-bit PCM
 * and write a standard PCM WAV so it plays everywhere (μ-law-in-WAV needs a fact
 * chunk and trips some players). Pure and dependency-free, so it's unit-tested.
 */

// The μ-law codec lives in the realtime audio module, which is where the phone
// path needs it too. Re-exported here so this file's existing importers keep
// working and there is only ever one table to get wrong.
export { ULAW_SILENCE, ulawByteToPcm16 } from "../realtime/audio/g711";
import { ULAW_SILENCE, ulawByteToPcm16 } from "../realtime/audio/g711";

function padTo(buf: Buffer, length: number): Buffer {
  if (buf.length >= length) return buf;
  return Buffer.concat([buf, Buffer.alloc(length - buf.length, ULAW_SILENCE)]);
}

/**
 * Interleave two μ-law channels into a 16-bit stereo PCM WAV. The shorter
 * channel is padded with silence so both run the full call length.
 */
export function encodeStereoWav(callerUlaw: Buffer, botUlaw: Buffer, sampleRate = 8000): Buffer {
  const samples = Math.max(callerUlaw.length, botUlaw.length);
  const left = padTo(callerUlaw, samples);
  const right = padTo(botUlaw, samples);

  const channels = 2;
  const bytesPerSample = 2;
  const dataLen = samples * channels * bytesPerSample;
  const out = Buffer.alloc(44 + dataLen);

  out.write("RIFF", 0, "ascii");
  out.writeUInt32LE(36 + dataLen, 4);
  out.write("WAVE", 8, "ascii");
  out.write("fmt ", 12, "ascii");
  out.writeUInt32LE(16, 16); // PCM fmt chunk size
  out.writeUInt16LE(1, 20); // audio format = PCM
  out.writeUInt16LE(channels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * channels * bytesPerSample, 28); // byte rate
  out.writeUInt16LE(channels * bytesPerSample, 32); // block align
  out.writeUInt16LE(bytesPerSample * 8, 34); // bits per sample
  out.write("data", 36, "ascii");
  out.writeUInt32LE(dataLen, 40);

  let off = 44;
  for (let i = 0; i < samples; i++) {
    out.writeInt16LE(ulawByteToPcm16(left[i]), off);
    out.writeInt16LE(ulawByteToPcm16(right[i]), off + 2);
    off += 4;
  }
  return out;
}
