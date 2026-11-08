import { encodeStereoWav } from "./wav";

/**
 * Accumulates a call's two audio directions and renders a stereo WAV.
 *
 * The caller's frames arrive from Twilio continuously in real time (silence
 * included), so the caller stream is the master clock. When a bot frame
 * arrives, the bot channel is first padded with silence up to the caller's
 * current length, so Elle's speech lands at roughly the right moment on the
 * shared timeline instead of being bunched together.
 *
 * Audio is held in memory for the duration of the call, then flushed once on
 * hang-up — fine for demo-length calls.
 */
export class CallRecorder {
  private caller: Buffer[] = [];
  private bot: Buffer[] = [];
  private callerLen = 0;
  private botLen = 0;

  pushCaller(ulaw: Buffer): void {
    this.caller.push(ulaw);
    this.callerLen += ulaw.length;
  }

  pushBot(ulaw: Buffer): void {
    if (this.botLen < this.callerLen) {
      const pad = this.callerLen - this.botLen;
      this.bot.push(Buffer.alloc(pad, 0xff)); // μ-law silence
      this.botLen += pad;
    }
    this.bot.push(ulaw);
    this.botLen += ulaw.length;
  }

  /** Call length in ms (G.711 μ-law @ 8 kHz = 8 bytes per ms). */
  get durationMs(): number {
    return Math.round(Math.max(this.callerLen, this.botLen) / 8);
  }

  get isEmpty(): boolean {
    return this.callerLen === 0 && this.botLen === 0;
  }

  toWav(): Buffer {
    return encodeStereoWav(Buffer.concat(this.caller), Buffer.concat(this.bot));
  }
}
