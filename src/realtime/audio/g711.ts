/**
 * G.711 μ-law codec and the streaming rate conversion between Twilio's phone
 * audio and a model that speaks linear PCM.
 *
 * Twilio Media Streams are μ-law at 8 kHz in both directions. Deepgram and the
 * OpenAI Realtime API both accept that format natively, so those paths are a
 * passthrough. Gemini Live does not: it wants 16-bit little-endian PCM at
 * 16 kHz going in and produces 16-bit PCM at 24 kHz coming out. Both ratios are
 * integers (2x up, 3x down), which keeps the conversion exact and cheap.
 *
 * The converters are STATEFUL because audio arrives in chunks and the chunk
 * boundaries are arbitrary. Interpolating without carrying the previous sample
 * puts a discontinuity at every boundary, and decimating without carrying the
 * remainder drops or duplicates samples when a chunk length is not a multiple
 * of the ratio. Both artifacts are audible as a periodic tick at exactly the
 * frame rate, which is a miserable bug to chase from a phone call, so the
 * carries are kept here and asserted offline.
 */

/** μ-law silence byte (encodes ~0). */
export const ULAW_SILENCE = 0xff;

const BIAS = 0x84;
const CLIP = 32635;

/** Expand one G.711 μ-law byte to a signed 16-bit PCM sample. */
export function ulawByteToPcm16(u: number): number {
  u = ~u & 0xff;
  const sign = u & 0x80;
  const exponent = (u >> 4) & 0x07;
  const mantissa = u & 0x0f;
  let sample = ((mantissa << 3) + BIAS) << exponent;
  sample -= BIAS;
  if (sign) sample = -sample;
  // `| 0` normalizes negative zero to zero. μ-law's 0x7F is a negative zero,
  // and -0 leaking out of here is the kind of value that reads as equal to 0
  // everywhere except the one comparison that uses Object.is semantics.
  return sample | 0;
}

/** Compress one signed 16-bit PCM sample to a G.711 μ-law byte. */
export function pcm16ToUlawByte(sample: number): number {
  let sign = 0;
  if (sample < 0) {
    sign = 0x80;
    sample = -sample;
  }
  if (sample > CLIP) sample = CLIP;
  sample += BIAS;

  // Exponent is the position of the highest set bit above the bias, which is
  // what gives μ-law its logarithmic response.
  let exponent = 7;
  for (let mask = 0x4000; (sample & mask) === 0 && exponent > 0; exponent--, mask >>= 1);

  const mantissa = (sample >> (exponent + 3)) & 0x0f;
  return ~(sign | (exponent << 4) | mantissa) & 0xff;
}

/** Decode a μ-law buffer to little-endian 16-bit PCM. */
export function ulawToPcm16(ulaw: Buffer): Buffer {
  const out = Buffer.alloc(ulaw.length * 2);
  for (let i = 0; i < ulaw.length; i++) out.writeInt16LE(ulawByteToPcm16(ulaw[i]), i * 2);
  return out;
}

/** Encode little-endian 16-bit PCM to μ-law. */
export function pcm16ToUlaw(pcm: Buffer): Buffer {
  const count = pcm.length >> 1;
  const out = Buffer.alloc(count);
  for (let i = 0; i < count; i++) out[i] = pcm16ToUlawByte(pcm.readInt16LE(i * 2));
  return out;
}

/**
 * Caller audio: μ-law 8 kHz to PCM16 16 kHz, doubling the rate by linear
 * interpolation. The interpolated sample sits halfway between the previous
 * sample and the current one, so `last` has to survive across chunks.
 */
export class CallerAudioUpsampler {
  private last = 0;

  /** @param ulaw one chunk of μ-law bytes at 8 kHz. */
  convert(ulaw: Buffer): Buffer {
    const out = Buffer.alloc(ulaw.length * 4); // 2 samples out per byte in, 2 bytes each
    let o = 0;
    for (let i = 0; i < ulaw.length; i++) {
      const cur = ulawByteToPcm16(ulaw[i]);
      out.writeInt16LE((this.last + cur) >> 1, o);
      out.writeInt16LE(cur, o + 2);
      this.last = cur;
      o += 4;
    }
    return out;
  }
}

/**
 * Model audio: PCM16 24 kHz to μ-law 8 kHz, decimating 3:1.
 *
 * Every group of three input samples is averaged rather than picked from,
 * because plain decimation folds everything above 4 kHz back down into the
 * speech band as aliasing. A 3-tap box average is a crude low-pass, but the
 * telephone band tops out at 3.4 kHz anyway, so the energy it fails to reject
 * is energy the caller was never going to hear.
 *
 * `pending` carries the samples left over when a chunk does not divide by 3.
 */
export class ModelAudioDownsampler {
  private pending: number[] = [];

  /** @param pcm one chunk of little-endian PCM16 at 24 kHz. */
  convert(pcm: Buffer): Buffer {
    const count = pcm.length >> 1;
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
      this.pending.push(pcm.readInt16LE(i * 2));
      if (this.pending.length === 3) {
        const avg = ((this.pending[0] + this.pending[1] + this.pending[2]) / 3) | 0;
        out.push(pcm16ToUlawByte(avg));
        this.pending.length = 0;
      }
    }
    return Buffer.from(out);
  }

  /** Flush a trailing partial group at end of turn so no audio is swallowed. */
  flush(): Buffer {
    if (this.pending.length === 0) return Buffer.alloc(0);
    const sum = this.pending.reduce((a, b) => a + b, 0);
    const avg = (sum / this.pending.length) | 0;
    this.pending.length = 0;
    return Buffer.from([pcm16ToUlawByte(avg)]);
  }
}
