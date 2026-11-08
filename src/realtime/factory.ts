import { config } from "../config";
import * as db from "../db";
import { RealtimeFactory } from "./types";
import { openAIRealtimeFactory } from "./openaiRealtime";
import { deepgramCascadeFactory } from "./deepgramCascade";
import { geminiLiveFactory } from "./geminiLive";

/**
 * Which realtime (barge-in) phone brain answers calls.
 *
 * The boot default comes from REALTIME_PROVIDER:
 *  - "deepgram" — the cascade: Deepgram STT + text LLM + Aura TTS. Cheap,
 *    tool-reliable, the production line.
 *  - "openai"   — hosted speech-to-speech interaction model. Natural duplex
 *    turn-taking, pricier per minute: DEMO mode.
 *  - "gemini"   — the same class of fused model, but on Vertex AI: no API key
 *    (Application Default Credentials), inside the GCP BAA, same bill. The
 *    preferred demo path when the point is to show duplex turn-taking without
 *    taking on another vendor.
 *
 * The owner can override it at RUNTIME (no redeploy) from the Telegram
 * channel; the override is persisted in the settings table and applies to the
 * NEXT call — in-flight calls keep the model they started with. A deploy or
 * instance recycle rehydrates the override from the DB when it survives, and
 * otherwise falls back to the env default. The override is read from the
 * store on every call rather than cached, so overlapping instances agree.
 *
 * An override may carry an EXPIRY. Demo mode costs roughly 5-10x per minute,
 * and the realistic failure is not a bad switch, it is a switch nobody
 * remembers to undo: the demo ends, the phone stays on the expensive brain,
 * and the bill arrives a month later. A demo switch therefore defaults to a
 * deadline, and the deadline is enforced on read rather than by a timer, so it
 * survives an instance recycle instead of dying with the process that set it.
 */

export type VoiceProvider = "deepgram" | "openai" | "gemini";

const SETTING_KEY = "voice_provider";
const EXPIRY_KEY = "voice_provider_expires_at";

const envDefault = (): VoiceProvider =>
  config.realtimeProvider === "deepgram" || config.realtimeProvider === "gemini"
    ? config.realtimeProvider
    : "openai";

/**
 * Read the override from the settings store, expiring it if its deadline has
 * passed. Deliberately NOT cached in memory: Cloud Run can be running more
 * than one instance (a revision rollover overlaps them by design), and an
 * in-memory copy means the instance that did not handle the switch keeps
 * answering on the old architecture until it recycles. The store is the single
 * source of truth, this runs once per call rather than per audio frame, and a
 * local SQLite read is measured in microseconds.
 */
function readOverride(): VoiceProvider | null {
  const saved = db.getSetting(SETTING_KEY);
  if (saved !== "deepgram" && saved !== "openai" && saved !== "gemini") return null;

  const expiresAt = Number(db.getSetting(EXPIRY_KEY) ?? "");
  if (Number.isFinite(expiresAt) && expiresAt > 0 && Date.now() >= expiresAt) {
    console.log(`[voice] demo override expired, reverting to ${envDefault()} (next call onward)`);
    db.setSetting(SETTING_KEY, "");
    db.setSetting(EXPIRY_KEY, "");
    return null;
  }
  return saved;
}

export function currentVoiceProvider(): VoiceProvider {
  return readOverride() ?? envDefault();
}

/** True when the current choice came from an owner override, not the env. */
export function voiceProviderOverridden(): boolean {
  return readOverride() !== null;
}

/** Minutes left on a timed override, or null when there is no deadline. */
export function voiceProviderMinutesLeft(): number | null {
  if (readOverride() === null) return null;
  const expiresAt = Number(db.getSetting(EXPIRY_KEY) ?? "");
  if (!Number.isFinite(expiresAt) || expiresAt <= 0) return null;
  return Math.max(1, Math.ceil((expiresAt - Date.now()) / 60_000));
}

/**
 * Set or clear the override. `ttlMinutes` puts a deadline on it; omit (or pass
 * 0) for an override that stands until it is cleared by hand.
 */
export function setVoiceProvider(p: VoiceProvider | null, ttlMinutes?: number): void {
  const expiresAt =
    p !== null && ttlMinutes && ttlMinutes > 0 ? Date.now() + ttlMinutes * 60_000 : null;
  db.setSetting(SETTING_KEY, p ?? "");
  db.setSetting(EXPIRY_KEY, expiresAt ? String(expiresAt) : "");
  console.log(
    `[voice] provider ${p ? `overridden to ${p}` : "override cleared"}` +
      `${expiresAt ? ` for ${ttlMinutes} min` : ""} (next call onward)`
  );
}

/** Resolved per call so an owner switch applies without a redeploy. */
export function currentRealtimeFactory(): RealtimeFactory {
  switch (currentVoiceProvider()) {
    case "deepgram":
      return deepgramCascadeFactory;
    case "gemini":
      return geminiLiveFactory;
    default:
      return openAIRealtimeFactory;
  }
}

/** Boot-time selection (kept for callers that want the env default only). */
export function selectRealtimeFactory(): RealtimeFactory {
  return currentRealtimeFactory();
}
