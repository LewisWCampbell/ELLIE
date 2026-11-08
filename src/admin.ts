import { Router, Request, Response, NextFunction } from "express";
import crypto from "crypto";
import { config } from "./config";
import * as db from "./db";
import { sharedCalendar } from "./booking/booking";
import { runCalendarCheck } from "./booking/calendarCheck";
import { sendDueReminders } from "./booking/reminders";
import { archiveStatus } from "./archive";
import { gapReport, formatGapReport, maybeSendDigest } from "./gaps";
import {
  currentVoiceProvider,
  voiceProviderMinutesLeft,
  voiceProviderOverridden,
} from "./realtime/factory";

/**
 * Owner-only views. Guarded by ADMIN_TOKEN (an X-Admin-Token header, preferred,
 * or a ?token= query param). With no token configured the routes 404, so caller
 * data is never exposed by default. This is a lightweight guard for a demo; put
 * real auth in front for production.
 */
export const adminRouter = Router();

function requireToken(req: Request, res: Response, next: NextFunction): void {
  const token = config.admin.token;
  if (!token) {
    res.status(404).send("Not found");
    return;
  }
  const provided =
    req.header("X-Admin-Token") || (typeof req.query.token === "string" ? req.query.token : "");
  // Constant-time comparison so response timing can't leak the token bytes.
  const a = Buffer.from(provided);
  const b = Buffer.from(token);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    res.status(403).send("Forbidden");
    return;
  }
  next();
}

function esc(v: unknown): string {
  return String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string
  );
}

/**
 * The booking calendar's end-to-end self-check, run BY THE SERVICE.
 *
 * The command-line version runs as whoever is at a terminal, which is the wrong
 * identity: Elle books as this service's account, and that is the account the
 * calendar has to be shared with. Checking from a laptop also runs into Google
 * refusing to grant sensitive scopes to the shared gcloud OAuth client, which
 * is a dead end unrelated to whether booking works. Running the check in here
 * sidesteps both: the credentials are already the right ones, and no consent
 * screen is involved.
 *
 * Creates one throwaway hold six months out and deletes it before returning.
 * Nobody is emailed; every write uses sendUpdates=none.
 */
adminRouter.get("/admin/calendar-check", requireToken, async (_req: Request, res: Response) => {
  const cal = sharedCalendar();
  if (!cal) {
    res
      .status(200)
      .type("text/plain")
      .send(
        "No calendar configured (BOOKING_CALENDAR_ID is unset), so booking captures\n" +
          "leads and creates no holds. Set it on the service and redeploy.\n"
      );
    return;
  }

  let result;
  try {
    result = await runCalendarCheck(cal, config.booking.timezone, config.booking.durationMinutes);
  } catch (e) {
    // The check handles expected failures itself and reports them as steps, so
    // reaching here means something unforeseen. Say what it was rather than
    // letting Express render a stack trace.
    res
      .status(500)
      .type("text/plain")
      .send(`The calendar check itself threw:\n\n  ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
    return;
  }

  const lines = [
    `calendar   ${config.booking.calendarId}`,
    `timezone   ${config.booking.timezone}`,
    `as         ${process.env.K_SERVICE ? "this service's own account" : "local credentials"}`,
    `revision   ${process.env.K_REVISION ?? "(not on Cloud Run)"}`,
    "",
    ...result.steps.map((s) => `  ${s.status.toUpperCase().padEnd(4)}  ${s.name.padEnd(28)} ${s.detail}`),
    "",
    result.ok ? "Booking calendar is working." : "Booking calendar is NOT working.",
  ];
  if (result.diagnosis) lines.push("", result.diagnosis);

  // 200 even when the calendar is broken. The REQUEST succeeded and the report
  // is the payload; a failing check is a finding, not a server error. Returning
  // 500 made curl --fail throw the report away and print nothing but the
  // status, which is the opposite of what a diagnostic is for.
  res.status(200).type("text/plain").send(lines.join("\n") + "\n");
});

/**
 * Run the reminder job. Token-guarded like every other admin route, so Cloud
 * Scheduler authenticates with the same header a person would use:
 *
 *   gcloud scheduler jobs create http elle-reminders \
 *     --schedule="0 17 * * *" --time-zone="America/Los_Angeles" \
 *     --uri="$URL/admin/run-reminders" --http-method=POST \
 *     --headers="X-Admin-Token=$(gcloud secrets versions access latest --secret=admin-token)"
 *
 * POST rather than GET: it changes things, and a scheduler retrying a GET that
 * a proxy decided to prefetch is a class of bug worth designing out.
 */
adminRouter.post("/admin/run-reminders", requireToken, async (_req: Request, res: Response) => {
  try {
    const run = await sendDueReminders();
    console.log("[reminders]", JSON.stringify(run));
    res.json(run);
  } catch (e) {
    console.error("[reminders] run failed:", e);
    res.status(500).json({ error: e instanceof Error ? e.message : String(e) });
  }
});

adminRouter.get("/admin/bookings.json", requireToken, (_req: Request, res: Response) => {
  res.json({ pending: db.getPendingBookings() });
});

// --- call history ------------------------------------------------------------

adminRouter.get("/admin/calls.json", requireToken, (_req: Request, res: Response) => {
  const calls = db.getRecentCalls(25).map((c) => ({
    ...c,
    transcript: db.getTranscripts(c.call_sid),
  }));
  res.json({ calls });
});

adminRouter.get("/admin/calls", requireToken, (_req: Request, res: Response) => {
  const calls = db.getRecentCalls(25);
  const body =
    calls.length === 0
      ? `<p class="empty">No calls recorded since this revision started. (Call history lives on the instance's ephemeral disk, so it resets on deploy. The Twilio Console keeps the permanent record.)</p>`
      : calls
          .map((c) => {
            const lines = db.getTranscripts(c.call_sid);
            const convo = lines.length
              ? lines
                  .map(
                    (l) =>
                      `<div class="line"><span class="who ${l.role === "assistant" ? "elle" : ""}">${
                        l.role === "assistant" ? "Elle" : "Caller"
                      }</span>${esc(l.content)}</div>`
                  )
                  .join("")
              : `<p class="muted">No transcript captured.</p>`;
            return `<details>
              <summary><span class="mono">${esc(c.started_at)}</span> &nbsp; <strong>${esc(
                c.phone_number
              )}</strong> &nbsp; <span class="muted">${c.ended_at ? "ended " + esc(c.ended_at) : "no end recorded"} · ${lines.length} turns</span></summary>
              <div class="convo">${convo}</div>
            </details>`;
          })
          .join("");

  res.type("html").send(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Recent calls</title>
<style>
  :root{--paper:#e7e6df;--ink:#17171b;--muted:#797882;--rule:rgba(23,23,27,.15);--rule2:rgba(23,23,27,.34);--accent:#1e37e6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--serif:"Charter","Iowan Old Style",Georgia,serif;}
  @media(prefers-color-scheme:dark){:root{--paper:#0d0d12;--ink:#ecebe4;--muted:#76757e;--rule:rgba(236,235,228,.15);--rule2:rgba(236,235,228,.3);--accent:#6d84ff;}}
  *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:16px;line-height:1.6}
  .wrap{max-width:900px;margin:0 auto;padding:3rem 1.5rem}
  .bar{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2);padding-bottom:.6rem}
  h1{font-family:var(--mono);font-size:2rem;letter-spacing:-.03em;margin:1.4rem 0 .4rem}
  .sub{color:var(--muted);margin:0 0 2rem}
  details{border-bottom:1px solid var(--rule);padding:.7rem 0}
  summary{cursor:pointer}
  .mono{font-family:var(--mono);font-size:.85em}.muted{color:var(--muted);font-size:.85em}
  .convo{padding:.8rem 0 .3rem 1rem;border-left:2px solid var(--rule2);margin-top:.6rem}
  .line{margin:.35rem 0}
  .who{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-right:.6rem}
  .who.elle{color:var(--accent)}
  .empty{color:var(--muted);font-family:var(--mono);font-size:.9rem}
</style></head><body><div class="wrap">
  <div class="bar"><span>LETSTALK // ELLE</span><span>CALLS · ${calls.length}</span></div>
  <h1>Recent calls</h1>
  <p class="sub">Who called and what was said, newest first. Click a call to read the conversation. SMS threads appear under their number too.</p>
  ${body}
</div></body></html>`);
});

// --- knowledge gaps ----------------------------------------------------------

function gapDays(req: Request): number {
  const n = Number(req.query.days ?? 7);
  return Number.isFinite(n) && n > 0 && n <= 365 ? n : 7;
}

adminRouter.get("/admin/gaps.json", requireToken, (req: Request, res: Response) => {
  res.json(gapReport(gapDays(req)));
});

/** Send the digest now, regardless of schedule. Useful for checking the wiring. */
adminRouter.post("/admin/send-digest", requireToken, async (_req: Request, res: Response) => {
  db.setSetting("gap_digest_at", new Date(0).toISOString());
  const result = await maybeSendDigest();
  res.json({ result, preview: formatGapReport(gapReport()) });
});

adminRouter.get("/admin/gaps", requireToken, (req: Request, res: Response) => {
  const days = gapDays(req);
  const r = gapReport(days);
  const rate = r.total ? Math.round((r.answered / r.total) * 100) : 0;

  const row = (c: { example: string; count: number; source: string | null }) =>
    `<div class="line"><span class="n">${c.count}&times;</span>${esc(c.example)}${
      c.source ? `<span class="src">${esc(c.source.replace(/\.md$/, ""))}</span>` : ""
    }</div>`;

  const body = !r.total
    ? `<p class="empty">Nothing logged in the last ${days} days.</p>`
    : `
    <h2>Could not answer</h2>
    <p class="sub">Each of these is a paragraph missing from <span class="mono">knowledge/</span>.</p>
    ${r.gaps.length ? r.gaps.map(row).join("") : `<p class="empty">Nothing unanswered. Either the corpus is complete or nobody asked anything hard.</p>`}

    <h2>Most asked</h2>
    ${r.topics.map(row).join("")}

    <h2>Answered from</h2>
    ${
      r.sources.length
        ? r.sources
            .map(
              (x) =>
                `<div class="line"><span class="n">${x.count}&times;</span>${esc(
                  x.source.replace(/\.md$/, "")
                )} <span class="muted">/ ${esc(x.heading)}</span></div>`
            )
            .join("")
        : `<p class="empty">No answered questions yet.</p>`
    }`;

  res.type("html").send(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Knowledge gaps</title>
<style>
  :root{--paper:#e7e6df;--ink:#17171b;--muted:#797882;--rule:rgba(23,23,27,.15);--rule2:rgba(23,23,27,.34);--accent:#1e37e6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--serif:"Charter","Iowan Old Style",Georgia,serif;}
  @media(prefers-color-scheme:dark){:root{--paper:#0d0d12;--ink:#ecebe4;--muted:#76757e;--rule:rgba(236,235,228,.15);--rule2:rgba(236,235,228,.3);--accent:#6d84ff;}}
  *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:16px;line-height:1.6}
  .wrap{max-width:900px;margin:0 auto;padding:3rem 1.5rem}
  .bar{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2);padding-bottom:.6rem}
  h1{font-family:var(--mono);font-size:2rem;letter-spacing:-.03em;margin:1.4rem 0 .4rem}
  h2{font-family:var(--mono);font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin:2.2rem 0 .5rem;border-bottom:1px solid var(--rule);padding-bottom:.4rem}
  .sub{color:var(--muted);margin:0 0 1.4rem}
  .line{display:flex;gap:.7rem;align-items:baseline;border-bottom:1px solid var(--rule);padding:.5rem 0}
  .n{font-family:var(--mono);font-size:.75rem;color:var(--accent);min-width:2.4rem;text-align:right}
  .src{margin-left:auto;font-family:var(--mono);font-size:.68rem;color:var(--muted);white-space:nowrap;padding-left:1rem}
  .mono{font-family:var(--mono);font-size:.85em}.muted{color:var(--muted);font-size:.85em}
  .empty{color:var(--muted);font-family:var(--mono);font-size:.9rem}
  .stat{font-family:var(--mono);font-size:.8rem;color:var(--muted)}
  .stat b{color:var(--ink);font-weight:600}
</style></head><body><div class="wrap">
  <div class="bar"><span>LETSTALK // ELLE</span><span>LAST ${days} DAYS</span></div>
  <h1>Knowledge gaps</h1>
  <p class="stat"><b>${r.total}</b> questions asked &middot; <b>${rate}%</b> answerable from the corpus &middot; ${
    Object.entries(r.byChannel)
      .map(([c, n]) => `${n} ${esc(c)}`)
      .join(", ") || "no channels yet"
  }</p>
  <p class="sub">Anonymous by construction: what was asked and whether it could be answered, with no number and no call attached.</p>
  ${body}
</div></body></html>`);
});

// --- feature status ----------------------------------------------------------

interface StatusRow {
  feature: string;
  state: "ok" | "warn" | "off";
  detail: string;
}

/**
 * Live configuration check. Presence checks read env-derived config; Calendar
 * and Telegram get REAL probes (a free/busy query and getWebhookInfo), because
 * "the env var is set" and "it actually works" have differed before.
 */
async function collectStatus(): Promise<StatusRow[]> {
  const rows: StatusRow[] = [];

  rows.push({
    feature: "LLM",
    state: config.llmProvider === "anthropic" && !config.anthropicApiKey ? "off" : "ok",
    detail: `provider=${config.llmProvider}, model=${config.claudeModel}` +
      (config.llmProvider === "anthropic" ? `, key ${config.anthropicApiKey ? "present" : "MISSING"}` : ""),
  });
  rows.push({
    feature: "Voice (Deepgram cascade)",
    state: config.deepgram.apiKey ? "ok" : "off",
    detail: `mode=${config.voiceMode}/${config.realtimeProvider}, key ${config.deepgram.apiKey ? "present" : "missing"}, ` +
      `stt=${config.deepgram.sttModel}, tts=${config.deepgram.ttsModel}, endpointing=${config.deepgram.endpointingMs}ms`,
  });
  rows.push({
    feature: "Active voice architecture",
    state: currentVoiceProvider() === "deepgram" ? "ok" : "warn",
    detail: `${
      currentVoiceProvider() === "deepgram"
        ? "cascade (production)"
        : currentVoiceProvider() === "gemini"
          ? "DEMO: duplex model on Vertex AI, several times the cascade's cost/min"
          : "DEMO: duplex model on OpenAI, ~5-10x cost/min"
    }` +
      `${voiceProviderOverridden() ? " — owner override" : " — deployed default"}` +
      `${voiceProviderMinutesLeft() ? `, auto-reverts in ${voiceProviderMinutesLeft()} min` : ""}`,
  });
  rows.push({
    feature: "Twilio webhook validation",
    state: config.twilioAuthToken ? "ok" : "warn",
    detail: config.twilioAuthToken ? "auth token present, signatures verified" : "no auth token: signature checks OFF",
  });
  rows.push({
    feature: "Outbound SMS (consent texts + outreach)",
    state: config.twilioAccountSid && config.twilioFromNumber ? "ok" : "off",
    detail: `account SID ${config.twilioAccountSid ? "present" : "missing"}, ` +
      `from number ${config.twilioFromNumber ? config.twilioFromNumber : "missing"}`,
  });

  // Calendar: config + live probe.
  if (!config.booking.calendarId) {
    rows.push({
      feature: "Google Calendar booking",
      state: "off",
      detail: "BOOKING_CALENDAR_ID not set: bookings are captured as leads only, no calendar holds",
    });
  } else {
    const cal = sharedCalendar();
    let state: StatusRow["state"] = "warn";
    let detail = `calendar=${config.booking.calendarId}, client not initialized`;
    if (cal) {
      try {
        const now = new Date();
        const end = new Date(now.getTime() + 30 * 60_000);
        const free = await cal.isFree(now.toISOString(), end.toISOString());
        state = "ok";
        detail = `calendar=${config.booking.calendarId}, live free/busy probe OK (next 30 min: ${free ? "free" : "busy"})`;
      } catch (e) {
        state = "warn";
        detail = `calendar=${config.booking.calendarId}, probe FAILED: ${String((e as Error)?.message ?? e).slice(0, 140)}`;
      }
    }
    rows.push({ feature: "Google Calendar booking", state, detail });
  }

  // Telegram: config + live webhook probe against the Bot API.
  if (!config.telegram.botToken) {
    rows.push({
      feature: "Telegram owner intake",
      state: "off",
      detail: "TELEGRAM_BOT_TOKEN not set: feature disabled",
    });
  } else {
    let state: StatusRow["state"] = "warn";
    let detail = "";
    try {
      const base = `https://api.telegram.org/bot${config.telegram.botToken}`;
      const [meRes, whRes] = await Promise.all([
        fetch(`${base}/getMe`, { signal: AbortSignal.timeout(6000) }),
        fetch(`${base}/getWebhookInfo`, { signal: AbortSignal.timeout(6000) }),
      ]);
      const me = (await meRes.json()) as { ok: boolean; result?: { username?: string } };
      const wh = (await whRes.json()) as {
        ok: boolean;
        result?: { url?: string; pending_update_count?: number; last_error_message?: string };
      };
      if (!me.ok) {
        detail = "bot token REJECTED by Telegram";
      } else {
        const url = wh.result?.url || "";
        const expected = `${config.publicUrl}/telegram`;
        const bits = [`bot @${me.result?.username ?? "?"}`];
        if (!url) bits.push("webhook NOT registered (run setWebhook)");
        else if (url !== expected) bits.push(`webhook points at ${url}, expected ${expected}`);
        else bits.push("webhook registered correctly");
        bits.push(`owner id ${config.telegram.ownerId ? "set" : "NOT set (setup mode)"}`);
        bits.push(`secret ${config.telegram.secret ? "set" : "NOT set"}`);
        if (wh.result?.pending_update_count) bits.push(`${wh.result.pending_update_count} pending updates`);
        if (wh.result?.last_error_message) bits.push(`last error: ${wh.result.last_error_message.slice(0, 80)}`);
        detail = bits.join("; ");
        state =
          me.ok && url === expected && config.telegram.ownerId && config.telegram.secret && !wh.result?.last_error_message
            ? "ok"
            : "warn";
      }
    } catch (e) {
      detail = `Telegram API probe failed: ${String((e as Error)?.message ?? e).slice(0, 100)}`;
    }
    rows.push({ feature: "Telegram owner intake", state, detail });
  }

  rows.push({
    feature: "Call recording",
    state: config.recording.enabled && config.recording.bucket ? "ok" : "off",
    detail: config.recording.enabled
      ? `enabled, bucket=${config.recording.bucket || "MISSING"}`
      : "disabled (consent notice not in greeting yet)",
  });

  // Call history is the one thing whose loss is silent: an empty call list
  // looks exactly like a quiet week. Say plainly which of the two it is.
  const arch = archiveStatus();
  rows.push({
    feature: "Call archive",
    state: arch.enabled ? (arch.lastError ? "warn" : "ok") : "warn",
    detail: arch.enabled
      ? `gs://${arch.bucket} · ${arch.note}` +
        (arch.pending ? ` · ${arch.pending} waiting to write` : "") +
        (arch.lastError ? ` · last error: ${arch.lastError}` : "")
      : "OFF. Call history lives on this container's disk and is lost on the next " +
        "deploy. Turn it on with ./tools/enable-archive.sh",
  });

  return rows;
}

adminRouter.get("/admin/status.json", requireToken, async (_req: Request, res: Response) => {
  res.json({ publicUrl: config.publicUrl, checks: await collectStatus() });
});

adminRouter.get("/admin/status", requireToken, async (_req: Request, res: Response) => {
  const rows = await collectStatus();
  const dot = { ok: "#1c7c4a", warn: "#b5871f", off: "#797882" } as const;
  const label = { ok: "READY", warn: "CHECK", off: "OFF" } as const;
  const body = `<table>
    <thead><tr><th></th><th>Feature</th><th>Detail</th></tr></thead>
    <tbody>${rows
      .map(
        (r) => `<tr>
          <td><span class="pill" style="background:${dot[r.state]}">${label[r.state]}</span></td>
          <td class="feat">${esc(r.feature)}</td>
          <td class="mono muted">${esc(r.detail)}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table>`;

  res.type("html").send(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Elle system status</title>
<style>
  :root{--paper:#e7e6df;--ink:#17171b;--muted:#797882;--rule:rgba(23,23,27,.15);--rule2:rgba(23,23,27,.34);--accent:#1e37e6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--serif:"Charter","Iowan Old Style",Georgia,serif;}
  @media(prefers-color-scheme:dark){:root{--paper:#0d0d12;--ink:#ecebe4;--muted:#76757e;--rule:rgba(236,235,228,.15);--rule2:rgba(236,235,228,.3);--accent:#6d84ff;}}
  *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:16px;line-height:1.6}
  .wrap{max-width:980px;margin:0 auto;padding:3rem 1.5rem}
  .bar{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2);padding-bottom:.6rem}
  h1{font-family:var(--mono);font-size:2rem;letter-spacing:-.03em;margin:1.4rem 0 .4rem}
  .sub{color:var(--muted);margin:0 0 2rem}
  table{border-collapse:collapse;width:100%;font-size:.95rem}
  th,td{text-align:left;padding:.7rem .8rem .7rem 0;border-bottom:1px solid var(--rule);vertical-align:top}
  th{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2)}
  .mono{font-family:var(--mono);font-size:.8em}.muted{color:var(--muted)}.feat{white-space:nowrap;font-weight:600}
  .pill{font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;color:#fff;padding:.25rem .5rem;border-radius:3px;white-space:nowrap}
</style></head><body><div class="wrap">
  <div class="bar"><span>LETSTALK // ELLE</span><span>${esc(new Date().toISOString())}</span></div>
  <h1>System status</h1>
  <p class="sub">Live configuration of this deployed revision. Calendar and Telegram rows are real probes, not just env checks. Refresh after each deploy.</p>
  ${body}
</div></body></html>`);
});

adminRouter.get("/admin/bookings", requireToken, (_req: Request, res: Response) => {
  const rows = db.getPendingBookings();
  const body =
    rows.length === 0
      ? `<p class="empty">No pending intro calls. You're all caught up.</p>`
      : `<table>
          <thead><tr><th>Requested time</th><th>Who</th><th>Topic</th><th>From</th><th></th></tr></thead>
          <tbody>${rows
            .map((b) => {
              const link = b.event_link
                ? `<a href="${esc(b.event_link)}" target="_blank" rel="noopener">Confirm in Calendar &rsaquo;</a>`
                : `<span class="muted">no calendar hold</span>`;
              return `<tr>
                <td class="mono">${esc(b.start_at)}</td>
                <td>${esc(b.name)}<br><span class="muted mono">${esc(b.email)}</span></td>
                <td>${esc(b.topic) || '<span class="muted">n/a</span>'}</td>
                <td class="mono muted">${esc(b.phone_number) || "web"}</td>
                <td>${link}</td>
              </tr>`;
            })
            .join("")}</tbody>
        </table>`;

  res.type("html").send(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Pending intro calls</title>
<style>
  :root{--paper:#e7e6df;--ink:#17171b;--muted:#797882;--rule:rgba(23,23,27,.15);--rule2:rgba(23,23,27,.34);--accent:#1e37e6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--serif:"Charter","Iowan Old Style",Georgia,serif;}
  @media(prefers-color-scheme:dark){:root{--paper:#0d0d12;--ink:#ecebe4;--muted:#76757e;--rule:rgba(236,235,228,.15);--rule2:rgba(236,235,228,.3);--accent:#6d84ff;}}
  *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:16px;line-height:1.6}
  .wrap{max-width:900px;margin:0 auto;padding:3rem 1.5rem}
  .bar{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2);padding-bottom:.6rem}
  h1{font-family:var(--mono);font-size:2rem;letter-spacing:-.03em;margin:1.4rem 0 .4rem}
  .sub{color:var(--muted);margin:0 0 2rem}
  table{border-collapse:collapse;width:100%;font-size:.95rem}
  th,td{text-align:left;padding:.7rem .8rem .7rem 0;border-bottom:1px solid var(--rule);vertical-align:top}
  th{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);border-bottom:1.5px solid var(--rule2)}
  .mono{font-family:var(--mono);font-size:.85em}.muted{color:var(--muted)}
  a{color:var(--accent);font-family:var(--mono);font-size:.8rem;text-decoration:none;border:1.5px solid var(--accent);padding:.35rem .6rem;white-space:nowrap}
  a:hover{background:var(--accent);color:var(--paper)}
  .empty{color:var(--muted);font-family:var(--mono);font-size:.9rem}
</style></head><body><div class="wrap">
  <div class="bar"><span>LETSTALK // ELLE</span><span>PENDING · ${rows.length}</span></div>
  <h1>Pending intro calls</h1>
  <p class="sub">Tentative holds Elle booked, waiting for you to confirm. Open each in Google Calendar to confirm the time and invite the caller.</p>
  ${body}
</div></body></html>`);
});
