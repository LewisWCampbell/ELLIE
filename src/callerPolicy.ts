import { config } from "./config";
import * as db from "./db";

/**
 * Repeat-caller escalation. Someone who dials over and over to run up the
 * per-minute cost gets progressively less service, keyed on their number:
 *
 *   normal  → full experience
 *   limited → the bot is told to keep it short; the call is time-capped
 *   blocked → a brief message, then hang up — the paid speech/LLM pipeline
 *             is never even started
 *
 * The tier is decided from how many calls the number has placed inside a
 * rolling window (config.abuse). Thresholds are inclusive of the current call,
 * so "escalates with each call" holds: the (limitedAfter+1)-th call is limited,
 * the (blockAfter+1)-th is blocked.
 */

export type CallerTier = "normal" | "limited" | "blocked";

export interface CallerAssessment {
  tier: CallerTier;
  /** 1-based index of this call within the window (prior calls + this one). */
  callNumber: number;
}

export function assessCaller(from: string): CallerAssessment {
  const priorCalls = db.countRecentCalls(from, config.abuse.windowHours);
  const callNumber = priorCalls + 1;
  if (callNumber > config.abuse.blockAfter) return { tier: "blocked", callNumber };
  if (callNumber > config.abuse.limitedAfter) return { tier: "limited", callNumber };
  return { tier: "normal", callNumber };
}

/** Spoken to a blocked caller right before hanging up. */
export function blockedMessage(): string {
  return (
    `You've reached ${config.assistantName} several times recently, so I'm not able to take ` +
    `another call right now. Please reach ${config.ownerName} by email instead and your ` +
    `message will get through. Thanks for understanding. Goodbye.`
  );
}

/** Appended to the system prompt for a rate-limited caller. */
export function limitedNote(): string {
  return (
    `\n\nNOTE: This caller has already called several times recently. Keep this call brief — ` +
    `answer concisely, don't volunteer extra detail, and gently steer them to follow up with ` +
    `${config.ownerName} by email. Wrap the call up within about a minute.`
  );
}
