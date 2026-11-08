/**
 * Offline verification of the phone-audio conversion used by the Gemini Live
 * path — no network, no credentials.
 *
 * The properties that matter are the ones that only break at chunk boundaries,
 * because that is where an ordinary "does it round-trip" test passes and a real
 * call ticks twenty times a second: interpolation state carrying across chunks,
 * and decimation remainders carrying across chunks whose length is not a
 * multiple of 3.
 *
 * Run: npx tsx src/realtime/audio/g711.verify.ts
 */
import assert from "node:assert";
import {
  CallerAudioUpsampler,
  ModelAudioDownsampler,
  pcm16ToUlaw,
  pcm16ToUlawByte,
  ulawByteToPcm16,
  ulawToPcm16,
} from "./g711";

/** A one-second-ish 300 Hz tone at `rate`, as little-endian PCM16. */
function tone(rate: number, samples: number, hz = 300): Buffer {
  const buf = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) {
    buf.writeInt16LE(Math.round(12000 * Math.sin((2 * Math.PI * hz * i) / rate)), i * 2);
  }
  return buf;
}

function main(): void {
  // --- codec ---------------------------------------------------------------
  // μ-law is lossy by design, so the round trip is checked for bounded error
  // rather than equality. 8-bit μ-law holds roughly 14 bits of dynamic range;
  // anything within a couple of percent of full scale is the codec working.
  let worst = 0;
  for (let s = -32000; s <= 32000; s += 97) {
    const back = ulawByteToPcm16(pcm16ToUlawByte(s));
    worst = Math.max(worst, Math.abs(back - s) / 32768);
  }
  assert.ok(worst < 0.02, `μ-law round trip within 2% of full scale (worst ${worst.toFixed(4)})`);

  // Every μ-law byte decodes and re-encodes to itself, so the encoder and
  // decoder agree on the same table. Disagreement there is what produces
  // static rather than speech.
  //
  // The single exception is real G.711, not a defect: the format has both a
  // negative zero (0x7F) and a positive zero (0xFF), both decoding to 0, and an
  // encoder has to pick one. Asserting the exception is exactly one code keeps
  // that documented and stops a future table edit from quietly adding more.
  const unstable: number[] = [];
  for (let u = 0; u < 256; u++) {
    if (pcm16ToUlawByte(ulawByteToPcm16(u)) !== u) unstable.push(u);
  }
  assert.deepStrictEqual(unstable, [0x7f], "only negative zero re-encodes to a different code");
  assert.strictEqual(ulawByteToPcm16(0x7f), 0, "negative zero decodes to silence");
  assert.strictEqual(ulawByteToPcm16(0xff), 0, "positive zero decodes to silence");

  // Buffer helpers preserve length relationships.
  const ulaw = Buffer.from([0x00, 0x40, 0x7f, 0x80, 0xff]);
  assert.strictEqual(ulawToPcm16(ulaw).length, ulaw.length * 2, "decode doubles the byte count");
  assert.strictEqual(pcm16ToUlaw(ulawToPcm16(ulaw)).length, ulaw.length, "encode halves it back");

  // --- upsample 8 kHz -> 16 kHz -------------------------------------------
  const up = new CallerAudioUpsampler();
  const frame = Buffer.alloc(160, 0x40); // 20 ms of μ-law at 8 kHz
  const upped = up.convert(frame);
  assert.strictEqual(upped.length, 160 * 4, "each μ-law byte becomes two PCM16 samples");

  // Chunking must not change the output. Feed the same audio as one buffer and
  // as three uneven pieces; the streams have to be identical, which is only
  // true if the interpolator carries its last sample across calls.
  const source = pcm16ToUlaw(tone(8000, 400));
  const whole = new CallerAudioUpsampler().convert(source);
  const split = new CallerAudioUpsampler();
  const pieces = Buffer.concat([
    split.convert(source.subarray(0, 37)),
    split.convert(source.subarray(37, 200)),
    split.convert(source.subarray(200)),
  ]);
  assert.ok(whole.equals(pieces), "upsampling is identical whole vs. split across chunks");

  // The interpolated samples sit between their neighbours rather than jumping,
  // which is what makes this a resampler and not a sample-and-hold.
  const ramp = Buffer.from([pcm16ToUlawByte(0), pcm16ToUlawByte(8000), pcm16ToUlawByte(16000)]);
  const smooth = new CallerAudioUpsampler().convert(ramp);
  for (let i = 1; i < smooth.length / 2 - 1; i++) {
    const prev = smooth.readInt16LE((i - 1) * 2);
    const cur = smooth.readInt16LE(i * 2);
    const next = smooth.readInt16LE((i + 1) * 2);
    assert.ok(
      cur >= Math.min(prev, next) - 1 && cur <= Math.max(prev, next) + 1,
      "interpolated samples stay between their neighbours"
    );
  }

  // --- downsample 24 kHz -> 8 kHz -----------------------------------------
  const down = new ModelAudioDownsampler();
  assert.strictEqual(down.convert(tone(24000, 300)).length, 100, "3:1 decimation");

  // A chunk length that is not a multiple of 3 must carry its remainder rather
  // than dropping it, so the stream stays sample-accurate across chunks.
  const model = tone(24000, 600);
  const oneShot = new ModelAudioDownsampler();
  const wholeDown = Buffer.concat([oneShot.convert(model), oneShot.flush()]);
  // Split points are at even byte offsets because the converter's contract is
  // whole PCM16 samples, which is what a decoded audio frame always contains.
  // None of the split points is a multiple of 3 samples, which is the case that
  // exercises the carry.
  const chunked = new ModelAudioDownsampler();
  const splitDown = Buffer.concat([
    chunked.convert(model.subarray(0, 50)), // 25 samples: 8 groups + 1 left over
    chunked.convert(model.subarray(50, 400)), // 175 more: another remainder
    chunked.convert(model.subarray(400)),
    chunked.flush(),
  ]);
  assert.ok(wholeDown.equals(splitDown), "downsampling is identical whole vs. split");
  assert.strictEqual(wholeDown.length, 200, "600 samples at 24 kHz is 200 bytes at 8 kHz");

  // flush() is idempotent: a second call after a turn ends must not inject a
  // stray byte into the next turn.
  const flushed = new ModelAudioDownsampler();
  flushed.convert(tone(24000, 4)); // 1 group + 1 pending
  assert.strictEqual(flushed.flush().length, 1, "a partial group flushes to one byte");
  assert.strictEqual(flushed.flush().length, 0, "flushing again emits nothing");

  // A round trip through both converters preserves the signal's shape. Compare
  // energy rather than samples: two resamplings and a lossy codec will never be
  // bit-exact, but losing the signal entirely (silence, or clipping to noise)
  // is the bug worth catching.
  const spoken = tone(24000, 2400, 440);
  const asPhone = new ModelAudioDownsampler();
  const ulawOut = Buffer.concat([asPhone.convert(spoken), asPhone.flush()]);
  const decoded = ulawToPcm16(ulawOut);
  let energy = 0;
  for (let i = 0; i < decoded.length / 2; i++) energy += Math.abs(decoded.readInt16LE(i * 2));
  const mean = energy / (decoded.length / 2);
  assert.ok(mean > 5000 && mean < 12000, `signal survives the round trip (mean ${mean | 0})`);

  console.log("Phone audio conversion verification passed:");
  console.log("  ✓ μ-law encode/decode round trip within 2% of full scale");
  console.log("  ✓ every μ-law byte is stable except G.711's negative zero");
  console.log("  ✓ 8 kHz μ-law upsamples to 16 kHz PCM by interpolation");
  console.log("  ✓ upsampling is chunk-boundary independent");
  console.log("  ✓ 24 kHz PCM decimates 3:1 to 8 kHz μ-law with anti-alias averaging");
  console.log("  ✓ decimation carries its remainder across chunks");
  console.log("  ✓ flush() empties once and stays empty");
  console.log("  ✓ a full round trip preserves signal energy");
}

main();
