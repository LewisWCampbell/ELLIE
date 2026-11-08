import { Router, Request, Response } from "express";
import WebSocket, { WebSocketServer } from "ws";
import {
  getSessionByWebToken,
  pairByCode,
  pushFactsSnapshot,
} from "./sessions";
import { answer, ChatTurn } from "./webchat";
import * as db from "./db";
import { resumeSections } from "./knowledge";
import { config } from "./config";

export const webRouter = Router();

/**
 * Small fixed-window per-IP rate limiter for the public API. In-memory is
 * fine on a single always-warm instance; it resets on deploy, which only
 * ever errs in the visitor's favor. Cloud Run fronts us, so the client IP
 * is the first hop of X-Forwarded-For.
 */
function clientIp(req: Request): string {
  // LAST entry, not first. X-Forwarded-For is client-appendable: Cloud Run
  // adds the real connecting IP to the END of whatever the client sent, so
  // trusting the first hop lets anyone mint fresh "IPs" per request, which
  // both bypasses every rate limit here and grows the bucket map without
  // bound. The last entry is the one our own infrastructure wrote.
  const xff = req.header("X-Forwarded-For");
  if (!xff) return req.socket.remoteAddress ?? "unknown";
  const hops = xff.split(",");
  return hops[hops.length - 1].trim() || "unknown";
}

const buckets = new Map<string, { count: number; resetAt: number }>();

function rateLimit(name: string, limit: number, windowMs: number) {
  return (req: Request, res: Response, next: () => void): void => {
    const key = `${name}:${clientIp(req)}`;
    const now = Date.now();
    const bucket = buckets.get(key);
    if (!bucket || now >= bucket.resetAt) {
      buckets.set(key, { count: 1, resetAt: now + windowMs });
      next();
      return;
    }
    bucket.count += 1;
    if (bucket.count > limit) {
      res.status(429).json({ error: "Too many requests. Give it a minute and try again." });
      return;
    }
    next();
  };
}

// Sweep stale buckets so the map can't grow unbounded.
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of buckets) if (now >= v.resetAt) buckets.delete(k);
}, 10 * 60 * 1000).unref();

/** The owner/assistant identity, for the pages' headers. */
webRouter.get("/api/profile", (_req: Request, res: Response) => {
  res.json({ ownerName: config.ownerName, assistantName: config.assistantName, website: config.publicUrl });
});

/** The resume/portfolio as display-ready sections (single source of truth = knowledge/). */
webRouter.get("/api/resume", (_req: Request, res: Response) => {
  res.json({ sections: resumeSections() });
});

/** Website chat — same persona + RAG as the phone bot, text in/out. */
webRouter.post("/api/chat", rateLimit("chat", 30, 10 * 60 * 1000), async (req: Request, res: Response) => {
  const message = String(req.body?.message ?? "").trim();
  if (!message) {
    res.status(400).json({ error: "Empty message." });
    return;
  }
  if (message.length > 2000) {
    res.status(400).json({ error: "That message is too long — keep it under 2,000 characters." });
    return;
  }
  const history: ChatTurn[] = Array.isArray(req.body?.history)
    ? req.body.history
        .filter((t: unknown): t is ChatTurn => {
          const turn = t as ChatTurn;
          return turn && (turn.role === "user" || turn.role === "assistant") && typeof turn.text === "string";
        })
        .slice(-10)
        // The turn COUNT was capped but each turn's length was not, so one
        // request could still carry ~100 KB of "history" straight into the
        // model. Same ceiling as a fresh message.
        .map((t: ChatTurn) => ({ role: t.role, text: t.text.slice(0, 2000) }))
    : [];
  try {
    const result = await answer(message, history);
    res.json(result);
  } catch (err) {
    console.error("Chat error:", err);
    res.status(500).json({ error: "The assistant had a hiccup. Please try again." });
  }
});

/** Browser submits the pairing code the caller heard on the phone. The rate
 *  limit makes brute-forcing the 6-digit code space impractical within a
 *  call's lifetime. */
webRouter.post("/api/pair", rateLimit("pair", 10, 60 * 1000), (req: Request, res: Response) => {
  const code = String(req.body?.code ?? "").replace(/\D/g, "");
  if (code.length !== 6) {
    res.status(400).json({ error: "Enter the 6-digit code from your call." });
    return;
  }
  const paired = pairByCode(code);
  if (!paired) {
    res.status(404).json({ error: "No active call found for that code. Check the digits and try again." });
    return;
  }
  res.json({ token: paired.token, callSid: paired.session.callSid });
});

/**
 * WebSocket server (noServer — mounted by index.ts) for paired browser sessions.
 * Handles upgrade requests on the /ws path.
 */
export function createPairingWss(): WebSocketServer {
  // A pairing client sends tiny JSON (typed text is capped at 500 chars);
  // the ws default max payload is 100 MiB, which is a memory grief vector
  // for anyone holding a valid token. 64 KiB is a hundred times headroom.
  const wss = new WebSocketServer({ noServer: true, maxPayload: 64 * 1024 });

  wss.on("connection", (socket: WebSocket, req) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    const token = url.searchParams.get("token") ?? "";
    const session = getSessionByWebToken(token);

    if (!session) {
      socket.close(4001, "Invalid or expired pairing token");
      return;
    }

    session.sockets.add(socket);

    // Replay everything that happened before this socket connected, so a
    // refreshed tab still shows the whole conversation.
    socket.send(
      JSON.stringify({
        type: "paired",
        callSid: session.callSid,
        callerNumber: session.callerNumber,
      })
    );
    for (const event of session.eventLog) {
      socket.send(JSON.stringify(event));
    }
    // The panel starts from the database, not from the replay: earlier calls
    // from this number left facts no event in this session ever mentioned.
    pushFactsSnapshot(session);

    socket.on("message", (raw) => {
      try {
        const msg = JSON.parse(String(raw));
        if (msg.type === "typed" && typeof msg.text === "string") {
          // A message typed into the paired page. On the realtime path it
          // becomes a conversation turn immediately (Elle answers out loud);
          // without injection it queues for the caller's next spoken turn.
          const text = msg.text.trim().slice(0, 500);
          if (!text) return;
          if (session.injectText) session.injectText(text);
          else session.pendingWebNotes.push(`On the website, the caller typed: "${text}".`);
        } else if (msg.type === "fact_edit" && typeof msg.fact === "string") {
          // The caller corrected something on the "what Elle knows" panel.
          // Scoped to this session's phone number inside the db call, so a
          // guessed id cannot touch another caller's notes.
          const id = Number(msg.id);
          const fact = msg.fact.trim().slice(0, 300);
          if (!Number.isInteger(id) || !fact || session.callerNumber === "unknown") return;
          if (db.updateFact(id, session.callerNumber, fact)) {
            pushFactsSnapshot(session);
            const note = `On the website, the caller corrected what you know about them to: "${fact}".`;
            if (session.injectText) session.injectText(note);
            else session.pendingWebNotes.push(note);
          }
        } else if (msg.type === "fact_delete") {
          const id = Number(msg.id);
          if (!Number.isInteger(id) || session.callerNumber === "unknown") return;
          if (db.deleteFact(id, session.callerNumber)) {
            pushFactsSnapshot(session);
            // Deletion is quiet on purpose: someone removing a note about
            // themselves does not want it read back out loud.
            session.pendingWebNotes.push("On the website, the caller removed one of the notes about them.");
          }
        } else if (msg.type === "choice" && typeof msg.option === "string") {
          const option = String(msg.option).trim().slice(0, 200);
          if (!option) return;
          // Same split: live injection (reads naturally in the transcript) when
          // the bridge registered it, else the cascaded path picks the note up
          // on the next voice turn.
          if (session.injectText) session.injectText(`I choose: ${option}`);
          else session.pendingWebNotes.push(`On the website, the caller clicked the option: "${option}".`);
        }
      } catch {
        /* ignore malformed messages */
      }
    });

    socket.on("close", () => {
      session.sockets.delete(socket);
    });
  });

  return wss;
}
