/**
 * Process entrypoint: wires every channel onto one Express app + one HTTP
 * server. Routes: /voice* (turn-based phone), /voice-realtime + /media WS
 * (barge-in phone), /sms, /telegram (owner intake), /api/* + /ws (website chat
 * and pairing), /admin/* (owner views), and static files. Fails fast at boot if
 * the selected voice mode's credentials are missing — a misconfigured deploy
 * should die loudly here, not at the first caller.
 */
import express from "express";
import http from "http";
import path from "path";
import { config } from "./config";
import { voiceRouter } from "./voice";
import { realtimeRouter } from "./realtime/route";
import { smsRouter } from "./sms";
import { telegramRouter } from "./telegram";
import { createMediaWss } from "./realtime/twilioBridge";
import { currentRealtimeFactory } from "./realtime/factory";
import { webRouter, createPairingWss } from "./web";
import { adminRouter } from "./admin";
import { sweepStaleSessions } from "./sessions";
import { sendDueReminders } from "./booking/reminders";
import { maybeSendDigest } from "./gaps";
import { startArchive, restoreFromArchive, flushAll, archiveEnabled } from "./archive";

/** Returns an error string if the configured text LLM provider is missing creds. */
function llmConfigError(): string | null {
  if (config.llmProvider === "anthropic" && !config.anthropicApiKey) {
    return "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.";
  }
  if (config.llmProvider === "vertex" && !config.vertex.projectId) {
    return (
      "LLM_PROVIDER=vertex requires VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) and Google " +
      "Application Default Credentials. See docs/vertex-baa.md."
    );
  }
  if (config.llmProvider === "openai" && (!config.openaiApiKey || !config.openaiModel)) {
    return "LLM_PROVIDER=openai requires OPENAI_API_KEY and OPENAI_MODEL (and usually OPENAI_BASE_URL).";
  }
  return null;
}

// Twilio request authentication is not optional in a deployment. The webhook
// signature checks and the /media HMAC token all key off TWILIO_AUTH_TOKEN and
// fail OPEN without it, which is the right behaviour for a laptop and the
// wrong one for a public endpoint: a deploy that loses the token binding would
// start accepting forged webhooks silently. Dying here makes that loud.
if (!config.twilioAuthToken && !config.allowUnsignedTwilio) {
  console.error(
    "TWILIO_AUTH_TOKEN is not set, so webhook signatures and the /media stream " +
      "token cannot be verified. Set the token (in production it comes from " +
      "Secret Manager via tools/deploy.sh), or set ALLOW_UNSIGNED_TWILIO=1 to " +
      "run without authentication for local development."
  );
  process.exit(1);
}
if (config.allowUnsignedTwilio && !config.twilioAuthToken) {
  console.warn(
    "[security] ALLOW_UNSIGNED_TWILIO=1: /voice, /sms and /media accept " +
      "unauthenticated requests. Local development only."
  );
}

// Validate the credentials the selected voice mode actually needs.
if (config.voiceMode === "realtime" && config.realtimeProvider === "openai") {
  if (!config.realtime.apiKey) {
    console.error("REALTIME_PROVIDER=openai requires OPENAI_REALTIME_API_KEY (or OPENAI_API_KEY).");
    process.exit(1);
  }
} else if (config.voiceMode === "realtime" && config.realtimeProvider === "deepgram") {
  // Cascade: Deepgram for speech + the text LLM for reasoning — both required.
  if (!config.deepgram.apiKey) {
    console.error("REALTIME_PROVIDER=deepgram requires DEEPGRAM_API_KEY. See docs/deepgram-cascade.md.");
    process.exit(1);
  }
  const llmErr = llmConfigError();
  if (llmErr) {
    console.error(llmErr);
    process.exit(1);
  }
} else {
  // Cascaded voice mode uses the text LLM directly.
  const llmErr = llmConfigError();
  if (llmErr) {
    console.error(llmErr);
    process.exit(1);
  }
}

const app = express();
app.use(express.urlencoded({ extended: false })); // Twilio webhooks are form-encoded
app.use(express.json());

// CORS for the API so a page on your main site can call chat/resume/profile.
app.use("/api", (req, res, next) => {
  const origins = config.corsOrigin.split(",").map((o) => o.trim());
  const reqOrigin = req.header("Origin");
  if (config.corsOrigin === "*") {
    res.header("Access-Control-Allow-Origin", "*");
  } else if (reqOrigin && origins.includes(reqOrigin)) {
    res.header("Access-Control-Allow-Origin", reqOrigin);
    res.header("Vary", "Origin");
  }
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.sendStatus(204);
    return;
  }
  next();
});

app.use(voiceRouter); // cascaded path: /voice, /voice/respond, /voice/status
app.use(realtimeRouter); // realtime path: /voice-realtime
app.use(smsRouter); // texting: /sms (booking updates + confirmations)
app.use(telegramRouter); // owner intake: /telegram (Lewis directs outreach)
app.use(webRouter); // pairing API
app.use(adminRouter); // owner-only views (/admin/bookings), token-guarded
// The web experience lives in ONE place: the ask page on the main site. These
// routes used to serve a separate, unbranded pairing UI on this origin, which
// split the visitor experience across two domains. They now redirect, so old
// links and bookmarks still land somewhere useful. /docs, /api, /ws, /media and
// the webhooks are unaffected.
const ASK_URL = `${config.siteUrl}${config.askPath}`;
for (const legacy of ["/", "/index.html", "/interactive.html", "/resume.html", "/cv.html"]) {
  app.get(legacy, (_req, res) => res.redirect(302, ASK_URL));
}

app.use(express.static(path.join(__dirname, "..", "public")));

app.get("/healthz", (_req, res) => {
  res.json({ ok: true, voiceMode: config.voiceMode });
});

const server = http.createServer(app);

// Two WebSocket endpoints share the HTTP server; route upgrades by path.
const pairingWss = createPairingWss(); // /ws  — browser pairing
const mediaWss = createMediaWss(currentRealtimeFactory); // /media — resolved per call (owner can switch at runtime)

/**
 * Browser pairing now runs cross-origin (the ask page is on the main site,
 * this service is the API), so the pairing socket needs its own origin check:
 * CORS never applies to WebSockets, and without this any page anywhere could
 * open one. Same allowlist as the HTTP API. Twilio's /media socket is exempt —
 * it comes from Twilio's backend, carries no Origin header, and is already
 * guarded by its own HMAC stream token.
 */
function originAllowed(req: http.IncomingMessage): boolean {
  if (config.corsOrigin === "*") return true;
  const origin = req.headers.origin;
  if (!origin) return false; // a browser always sends one; a script may not
  return config.corsOrigin
    .split(",")
    .map((o) => o.trim())
    .includes(origin);
}

server.on("upgrade", (req, socket, head) => {
  let pathname = "/";
  try {
    pathname = new URL(req.url ?? "/", "http://localhost").pathname;
  } catch {
    /* fall through to destroy */
  }
  if (pathname === "/ws" && !originAllowed(req)) {
    console.warn(`[ws] rejected pairing upgrade from origin ${req.headers.origin ?? "(none)"}`);
    socket.destroy();
    return;
  }
  if (pathname === "/ws") {
    pairingWss.handleUpgrade(req, socket, head, (ws) => pairingWss.emit("connection", ws, req));
  } else if (pathname === "/media") {
    mediaWss.handleUpgrade(req, socket, head, (ws) => mediaWss.emit("connection", ws, req));
  } else {
    socket.destroy();
  }
});

setInterval(sweepStaleSessions, 15 * 60 * 1000).unref();

/**
 * Reminder sweep. Cloud Scheduler hitting /admin/run-reminders is the intended
 * trigger, but this service already holds a warm instance, and a feature that
 * only works once someone remembers to create a scheduler job is a feature
 * that silently does not work. The database claim makes the two safe to run
 * together: whichever gets there first sends, the other finds nothing due.
 */
setInterval(() => {
  void sendDueReminders().then((run) => {
    if (run.sent || run.failed) console.log("[reminders]", JSON.stringify(run));
  });
  // Rides the same hour: the digest decides for itself whether it is due.
  void maybeSendDigest().then((r) => {
    if (r === "sent") console.log("[gaps] weekly digest sent");
  });
}, 60 * 60 * 1000).unref();

// Begin copying calls to durable storage before the first one can arrive.
startArchive();

server.listen(config.port, () => {
  console.log(`LetsTalk listening on port ${config.port} (voice mode: ${config.voiceMode})`);
  console.log(`Public URL: ${config.publicUrl}`);
  if (config.voiceMode === "realtime") {
    console.log(`Realtime provider: ${config.realtimeProvider}`);
    console.log(`Twilio voice webhook: ${config.publicUrl}/voice-realtime`);
    console.log(`Media stream WS:      ${config.publicUrl.replace(/^http/, "ws")}/media`);
    console.log(`Twilio SMS webhook:   ${config.publicUrl}/sms`);
    console.log(`Web experience:       ${ASK_URL}  (chat + pairing)`);
    if (config.telegram.botToken) {
      console.log(`Telegram webhook:     ${config.publicUrl}/telegram`);
    }
  } else {
    console.log(`Twilio voice webhook: ${config.publicUrl}/voice`);
    console.log(`Twilio status callback: ${config.publicUrl}/voice/status`);
  }

  // Replay past calls AFTER the port is open. History is not needed to answer
  // a phone call, so a slow or unreachable bucket must not delay accepting
  // one, and a failure here leaves the service running with an empty history
  // rather than not running at all.
  void restoreFromArchive();
});

/**
 * Cloud Run sends SIGTERM and then waits before killing the container, which
 * is the one chance to write out a call that is mid-flight. Without this, a
 * deploy during a conversation loses the part of it since the last flush.
 */
let shuttingDown = false;
for (const signal of ["SIGTERM", "SIGINT"] as const) {
  process.on(signal, () => {
    if (shuttingDown) return;
    shuttingDown = true;
    const done = (): void => process.exit(0);
    if (!archiveEnabled()) return done();
    console.log("[archive] flushing before shutdown");
    // Bounded: a hung upload must not hold the container open until it is
    // killed anyway, which would lose the flush AND delay the deploy.
    const bail = setTimeout(done, 4000);
    bail.unref?.();
    void flushAll().finally(done);
  });
}
