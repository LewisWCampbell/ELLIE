import crypto from "crypto";
import type { WebSocket } from "ws";
import { Fact, getFactsForCaller } from "./db";
import type { NeutralMessage } from "./llm/types";

/** An event pushed to a paired browser session over WebSocket. */
export type WebEvent =
  | { type: "paired"; callSid: string; callerNumber: string }
  | { type: "transcript"; role: "caller" | "assistant"; text: string }
  | { type: "image"; url: string; caption: string }
  | { type: "show"; kind: "image" | "page" | "link"; url: string; caption: string }
  | { type: "choices"; question: string; options: string[] }
  | { type: "fact"; category: string; fact: string }
  | { type: "facts"; facts: Array<{ id: number; category: string; fact: string }> }
  | { type: "call_ended" };

export interface CallSession {
  callSid: string;
  callerNumber: string;
  pairingCode: string;
  /** Token minted when a browser successfully pairs with this call. */
  webToken: string | null;
  /** Connected browser sockets (a user might open more than one tab). */
  sockets: Set<WebSocket>;
  /** Full conversation history for this call (provider-neutral). */
  messages: NeutralMessage[];
  /** Facts known about this caller at call start (for the system prompt). */
  knownFacts: Fact[];
  /** Notes queued from the web session (e.g. option clicks) to feed into the next turn. */
  pendingWebNotes: string[];
  /**
   * Live text injection into the call, registered by the realtime bridge. A
   * typed message from the paired browser becomes a conversation turn at once —
   * Elle answers out loud on the phone — instead of waiting for the caller's
   * next spoken turn. Absent on the cascaded path (which polls pendingWebNotes).
   */
  injectText?: (text: string) => void;
  /** Set when the model calls end_call; the voice route then hangs up. */
  shouldHangup: boolean;
  /** Events sent before the browser connected, replayed on socket attach. */
  eventLog: WebEvent[];
  createdAt: number;
}

const sessionsByCallSid = new Map<string, CallSession>();
const sessionsByPairingCode = new Map<string, CallSession>();
const sessionsByWebToken = new Map<string, CallSession>();

function generatePairingCode(): string {
  // 6 digits, avoiding collisions with active calls.
  for (let attempt = 0; attempt < 50; attempt++) {
    const code = String(crypto.randomInt(0, 1_000_000)).padStart(6, "0");
    if (!sessionsByPairingCode.has(code)) return code;
  }
  throw new Error("Could not allocate a unique pairing code");
}

export function createSession(callSid: string, callerNumber: string, knownFacts: Fact[]): CallSession {
  const existing = sessionsByCallSid.get(callSid);
  if (existing) return existing;

  const session: CallSession = {
    callSid,
    callerNumber,
    pairingCode: generatePairingCode(),
    webToken: null,
    sockets: new Set(),
    messages: [],
    knownFacts,
    pendingWebNotes: [],
    shouldHangup: false,
    eventLog: [],
    createdAt: Date.now(),
  };
  sessionsByCallSid.set(callSid, session);
  sessionsByPairingCode.set(session.pairingCode, session);
  return session;
}

export function getSessionByCallSid(callSid: string): CallSession | undefined {
  return sessionsByCallSid.get(callSid);
}

/** Pair a browser with a call. Returns a web token, or null if the code is unknown. */
export function pairByCode(code: string): { token: string; session: CallSession } | null {
  const session = sessionsByPairingCode.get(code.trim());
  if (!session) return null;
  if (!session.webToken) {
    session.webToken = crypto.randomBytes(24).toString("hex");
    sessionsByWebToken.set(session.webToken, session);
    session.pendingWebNotes.push(
      "The caller just paired their web browser with this call. You can now use show_work and show_choices — prefer the live, interactive pages from the visuals list."
    );
  }
  return { token: session.webToken, session };
}

export function getSessionByWebToken(token: string): CallSession | undefined {
  return sessionsByWebToken.get(token);
}

export function isPaired(session: CallSession): boolean {
  return session.webToken !== null;
}

/** Send an event to every browser paired with this call (and log it for replay). */
/**
 * Send the paired page the full current list of what Elle knows about this
 * caller. A snapshot rather than increments, because the page lets the caller
 * EDIT the list: after a correction the only truthful display is whatever the
 * database now holds, ids included, and replaying "added X" events can never
 * reconstruct that. Sent on pair, after every save, and after every edit.
 */
/**
 * Decide how the paired page should present a shown URL.
 *
 * "page" means an iframe, and only two hosts qualify: the portfolio site
 * itself and Hugging Face's *.hf.space embed hosts, both of which are built to
 * be framed. Everything else becomes a "link" card, because a foreign site
 * that sends X-Frame-Options renders as a silent white rectangle, which reads
 * as the demo being broken rather than the embed being refused.
 */
export function classifyShowUrl(url: string, siteUrl: string): "image" | "page" | "link" {
  if (/\.(png|jpe?g|gif|webp|svg)(\?|$)/i.test(url)) return "image";
  try {
    const u = new URL(url);
    const site = new URL(siteUrl);
    if (u.hostname === site.hostname) return "page";
    if (/\.hf\.space$/i.test(u.hostname)) return "page";
  } catch {
    return "link";
  }
  return "link";
}

/** Calls currently in progress, for the capacity gate on the voice route. */
export function liveCallCount(): number {
  return sessionsByCallSid.size;
}

export function pushFactsSnapshot(session: CallSession): void {
  if (!session.callerNumber || session.callerNumber === "unknown") return;
  const facts = getFactsForCaller(session.callerNumber).map((f) => ({
    id: f.id,
    category: f.category,
    fact: f.fact,
  }));
  pushWebEvent(session, { type: "facts", facts });
}

export function pushWebEvent(session: CallSession, event: WebEvent): void {
  session.eventLog.push(event);
  const payload = JSON.stringify(event);
  for (const socket of session.sockets) {
    if (socket.readyState === socket.OPEN) {
      socket.send(payload);
    }
  }
}

export function endSession(callSid: string): void {
  const session = sessionsByCallSid.get(callSid);
  if (!session) return;
  pushWebEvent(session, { type: "call_ended" });
  for (const socket of session.sockets) {
    try {
      socket.close();
    } catch {
      /* already closed */
    }
  }
  sessionsByCallSid.delete(callSid);
  sessionsByPairingCode.delete(session.pairingCode);
  if (session.webToken) sessionsByWebToken.delete(session.webToken);
}

/** Drop sessions older than 4 hours in case Twilio status callbacks are missed. */
export function sweepStaleSessions(): void {
  const cutoff = Date.now() - 4 * 60 * 60 * 1000;
  for (const [sid, session] of sessionsByCallSid) {
    if (session.createdAt < cutoff) endSession(sid);
  }
}
