import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

const profilePath = process.env.PROFILE_PATH ?? "profile.md";

/**
 * Read a CREDENTIAL from the environment, trimming surrounding whitespace.
 *
 * Secret Manager stores exactly the bytes it was given, and Cloud Run injects
 * them verbatim — so a secret created with `echo` instead of `printf` carries a
 * trailing newline. Every comparison against it then fails silently: the
 * Telegram webhook drops every update, Twilio signatures never validate, an API
 * key 401s. Nothing in a credential is ever meaningfully whitespace, so trim
 * once here rather than at each use site.
 */
function secretEnv(...names: string[]): string {
  for (const n of names) {
    const v = process.env[n];
    if (v != null && v.trim()) return v.trim();
  }
  return "";
}

/**
 * Bot mode:
 *  - "interview": the bot speaks on behalf of the person described in the
 *    profile file, answering callers' questions about them (recruiter mode).
 *  - "learn": the bot interviews the caller and records facts about them.
 * Default: "interview" when the profile file exists, otherwise "learn".
 */
function resolveMode(): "interview" | "learn" {
  const explicit = process.env.BOT_MODE;
  if (explicit === "interview" || explicit === "learn") return explicit;
  return fs.existsSync(profilePath) ? "interview" : "learn";
}

export const config = {
  port: parseInt(process.env.PORT ?? "3000", 10),

  /**
   * Public base URL of this server (e.g. an ngrok URL or your domain).
   * Used to build the pairing link the web client shows and for Twilio
   * signature validation. Example: https://letstalk.example.com
   */
  publicUrl: (process.env.PUBLIC_URL ?? "http://localhost:3000").replace(/\/+$/, ""),

  /**
   * The HUMAN-facing website, as distinct from publicUrl above (which is this
   * service's own origin, used for webhooks and media streams). Visitors and
   * callers are always pointed here: it is where the branded chat and the
   * pairing screen live. Defaults to publicUrl so a standalone deploy still
   * self-references correctly. No trailing slash.
   */
  siteUrl: (process.env.SITE_URL ?? process.env.PUBLIC_URL ?? "http://localhost:3000").replace(/\/+$/, ""),

  /** The single page carrying both the chat and the pairing screen. */
  askPath: process.env.ASK_PATH ?? "/ask.html",

  /**
   * Voice transport:
   *  - "cascaded": Twilio Gather (speech-to-text) → LLM → Say (TTS). Turn-based,
   *    cheap, no barge-in. This is the original path.
   *  - "realtime": Twilio Media Streams ↔ OpenAI Realtime speech-to-speech.
   *    Native barge-in (caller can talk over the bot and it adapts).
   */
  voiceMode: (process.env.VOICE_MODE ?? "cascaded") as "cascaded" | "realtime",

  /**
   * Which realtime (barge-in) phone brain to use when voiceMode = "realtime":
   *  - "openai":   OpenAI Realtime speech-to-speech (native barge-in, priciest).
   *  - "deepgram": Deepgram STT + text LLM (the llmProvider, e.g. Claude on
   *    Vertex) + Deepgram Aura TTS, with VAD barge-in. Much cheaper per minute.
   */
  realtimeProvider: (process.env.REALTIME_PROVIDER ?? "openai") as
    | "openai"
    | "deepgram"
    | "gemini",

  /** OpenAI Realtime API settings (realtimeProvider = "openai"). */
  realtime: {
    apiKey: secretEnv("OPENAI_REALTIME_API_KEY", "OPENAI_API_KEY"),
    model: process.env.OPENAI_REALTIME_MODEL ?? "gpt-realtime",
    voice: process.env.REALTIME_VOICE ?? "alloy",
  },

  /**
   * Gemini Live on Vertex AI (realtimeProvider = "gemini"): the fused
   * speech-to-speech option that runs inside Google Cloud. No API key exists
   * for this path on purpose — auth is Application Default Credentials, so on
   * Cloud Run it is the service account already attached, and the calls fall
   * under the same GCP BAA as the compute. Project id is shared with the Vertex
   * LLM settings above; the Live API is not available in every region, so its
   * location is configured separately from VERTEX_REGION.
   */
  geminiLive: {
    model: process.env.GEMINI_LIVE_MODEL ?? "gemini-live-2.5-flash-preview-native-audio",
    location: process.env.GEMINI_LIVE_LOCATION ?? "us-central1",
    voice: process.env.GEMINI_LIVE_VOICE ?? "Aoede",
  },

  /**
   * Deepgram settings (realtimeProvider = "deepgram"). One key covers both the
   * streaming STT and the Aura TTS. Both legs run at G.711 μ-law 8 kHz to match
   * Twilio with no transcoding.
   */
  deepgram: {
    apiKey: secretEnv("DEEPGRAM_API_KEY"),
    sttModel: process.env.DEEPGRAM_STT_MODEL ?? "nova-2-phonecall",
    /**
     * STT model for bilingual calls. Nova-3 with `language=multi` transcribes
     * English/Spanish code-switching in ONE stream — the caller can change
     * language mid-call (even mid-sentence) and each utterance is tagged with
     * the language it was spoken in. Used only when languages.supported lists
     * more than one language; single-language deploys keep sttModel above.
     */
    sttMultiModel: process.env.DEEPGRAM_STT_MULTI_MODEL ?? "nova-3",
    /**
     * Elle's English voice. aura-2-thalia-en is a bright, high-energy American
     * female voice from the Aura-2 catalog: it sits well above the phone band's
     * muddy low end, which matters more here than it would on a laptop speaker,
     * since G.711 rolls off everything under ~300 Hz anyway. Other bright
     * options, roughly brightest first: aura-2-electra-en (energetic, dynamic),
     * aura-stella-en (fast, casual, high energy), aura-2-amalthea-en (warm and
     * approachable rather than peppy). tools/preview-voices.sh renders the
     * greeting through each so they can be compared by ear before switching.
     */
    ttsModel: process.env.DEEPGRAM_TTS_MODEL ?? "aura-2-thalia-en",
    /** Elle's Spanish voice (Aura-2 Spanish catalog), used when a caller speaks
     *  Spanish. Alternatives: aura-2-estrella-es, aura-2-diana-es, aura-2-javier-es. */
    ttsModelEs: process.env.DEEPGRAM_TTS_MODEL_ES ?? "aura-2-celeste-es",
    /**
     * How long a caller can pause before Elle decides they're done talking, in
     * ms of trailing silence. Higher = more patient (won't cut off someone
     * mid-thought) but adds that delay before she replies. Default: 1 second.
     */
    endpointingMs: process.env.DEEPGRAM_ENDPOINTING_MS ?? "1000",
    /**
     * Barge-in sensitivity. Deepgram's raw VAD fires on any sound (a breath, a
     * cough, line noise, or an echo of Elle's own voice), which was cutting her
     * off mid-sentence. Two guards make interruption require real intent:
     *  - `bargeInMinChars`: minimum characters of *recognized speech* before it
     *    counts as an interruption (noise rarely transcribes into words). Higher
     *    = harder to interrupt.
     *  - `bargeInGraceMs`: ignore barge-in for this many ms after Elle starts a
     *    reply, so an echo of her opening words can't cut her off instantly.
     */
    bargeInMinChars: parseInt(process.env.DEEPGRAM_BARGE_IN_MIN_CHARS ?? "6", 10),
    bargeInGraceMs: parseInt(process.env.DEEPGRAM_BARGE_IN_GRACE_MS ?? "1200", 10),
  },

  /**
   * LLM provider for the text brain (cascaded voice + website chat):
   *  - "anthropic" (Claude via api.anthropic.com, default)
   *  - "vertex"    (Claude via Google Vertex AI — inside your GCP BAA)
   *  - "openai"    (any OpenAI-compatible endpoint: Hugging Face, Groq, ...)
   */
  llmProvider: (process.env.LLM_PROVIDER ?? "anthropic") as "anthropic" | "vertex" | "openai",

  anthropicApiKey: secretEnv("ANTHROPIC_API_KEY"),
  claudeModel: process.env.CLAUDE_MODEL ?? "claude-haiku-4-5",

  /**
   * Claude on Google Vertex AI (LLM_PROVIDER=vertex). Auth is Application
   * Default Credentials (the machine's service account on Cloud Run/GCE, or
   * `gcloud auth application-default login` locally) — no API key is stored.
   * VERTEX_MODEL must be the exact Model Garden id available in your region,
   * e.g. "claude-sonnet-4-5@20250929". See docs/vertex-baa.md.
   */
  vertex: {
    projectId: process.env.VERTEX_PROJECT_ID ?? process.env.GOOGLE_CLOUD_PROJECT ?? "",
    region: process.env.VERTEX_REGION ?? "us-east5",
    model: process.env.VERTEX_MODEL ?? "claude-sonnet-4-5@20250929",
  },

  openaiBaseUrl: process.env.OPENAI_BASE_URL ?? "https://router.huggingface.co/v1",
  openaiApiKey: process.env.OPENAI_API_KEY ?? "",
  openaiModel: process.env.OPENAI_MODEL ?? "",

  /**
   * Languages Elle can hold a conversation in. More than one entry turns on
   * bilingual mode end to end: code-switching STT (deepgram.sttMultiModel), a
   * per-language TTS voice (ttsModel / ttsModelEs), a mirror-the-caller rule in
   * the persona, and a short Spanish invitation in the phone greeting. Set
   * LANGUAGES=en to run the original English-only stack. The text LLM needs no
   * configuration — Claude is natively multilingual.
   */
  languages: {
    supported: (process.env.LANGUAGES ?? "en,es")
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  },

  /** Markdown file describing the person the bot represents in interview mode. */
  profilePath,
  botMode: resolveMode(),

  /** The person the assistant represents (used in the persona/template). */
  ownerName: process.env.PROFILE_NAME ?? "the owner",

  /**
   * The owner's first name, for speech. A full name works in print and turns
   * stilted out loud the second time it is said: the TTS gives each word its
   * own weight, and "ask me anything about Lewis Campbell and his work" lands
   * like a form being read back. Only a real configured name is split; the
   * "the owner" fallback would otherwise become "the".
   */
  ownerFirstName: process.env.PROFILE_NAME
    ? (process.env.PROFILE_NAME.trim().split(/\s+/)[0] ?? process.env.PROFILE_NAME)
    : "the owner",

  /** The assistant's own name/persona. */
  assistantName: process.env.ASSISTANT_NAME ?? "Elle",

  /**
   * The legal entity registered as the A2P 10DLC brand.
   *
   * Carriers require the party responsible for a messaging program to be
   * identifiable, and a registration whose brand nobody can find in the
   * messages is rejected as a different business borrowing the number. It
   * appears only where that identification is required, which is the HELP
   * reply and the enrollment confirmation, and never in ordinary conversation:
   * a recipient is talking to Elle about an appointment, not to a company.
   *
   * Must match the Trust Hub brand character for character. Empty leaves every
   * message exactly as it was.
   */
  brandName: process.env.BRAND_NAME ?? "",

  /** Directory of markdown files that form the RAG knowledge base (resume, projects, skills). */
  knowledgeDir: process.env.KNOWLEDGE_DIR ?? "knowledge",

  /**
   * Allowed origin for the /api/* endpoints, so a page on your main site
   * (e.g. https://lewiswcampbell.com) can call the chat/resume API. Comma-
   * separate multiple origins, or "*" to allow any (fine for a public demo).
   */
  corsOrigin: process.env.CORS_ORIGIN ?? "*",

  /**
   * Twilio auth token. When set, incoming webhook requests are verified
   * with Twilio's request signature. Leave empty to skip validation
   * (local development only).
   */
  twilioAuthToken: secretEnv("TWILIO_AUTH_TOKEN"),

  /**
   * Explicit opt-out of Twilio request authentication, for local development
   * where no auth token exists. Three checks key off the token: webhook
   * signature validation on /voice and /sms, and the HMAC token that gates the
   * /media WebSocket. Without the token those checks cannot run, and silently
   * skipping them would mean a deploy that loses the token binding starts
   * accepting forged webhooks without anyone noticing. So a missing token is
   * fatal at boot unless this flag says, in so many words, that unsigned
   * traffic is intended.
   */
  allowUnsignedTwilio: process.env.ALLOW_UNSIGNED_TWILIO === "1",

  /**
   * Outbound SMS (the verbal-opt-in confirmation text). Both must be set for
   * outbound sends; everything else (webhook replies) works without them.
   */
  twilioAccountSid: process.env.TWILIO_ACCOUNT_SID ?? "",
  twilioFromNumber: process.env.TWILIO_FROM_NUMBER ?? "",

  /** Voice used for text-to-speech in <Say>. */
  ttsVoice: process.env.TTS_VOICE ?? "Google.en-US-Chirp3-HD-Aoede",

  sqlitePath: process.env.SQLITE_PATH ?? "letstalk.db",

  /**
   * Cloud Storage bucket for the durable call archive. The SQLite file above
   * lives on the container's disk, which Cloud Run throws away on every deploy
   * and every instance replacement, so without this the record of who called
   * and what was said lasts only as long as the current container. With it,
   * each call is written to the bucket as it happens and replayed on startup.
   * Empty means the archive is off and history is ephemeral.
   */
  callArchiveBucket: process.env.CALL_ARCHIVE_BUCKET ?? "",

  /**
   * How many calls may be in progress at once. Past this the next caller is
   * offered a callback instead of being connected.
   *
   * Silence is the failure mode this replaces: with capacity exhausted the
   * caller previously heard nothing at all and hung up believing the number
   * was broken. A stated "we're busy, text me and I'll get back to you" keeps
   * the lead and tells the truth.
   */
  maxConcurrentCalls: Number(process.env.MAX_CONCURRENT_CALLS ?? 20),

  /** How many days of archived calls to replay on startup. */
  callArchiveRestoreDays: Number(process.env.CALL_ARCHIVE_RESTORE_DAYS ?? 30),

  /**
   * The knowledge-gap report: an anonymous log of what people ask and whether
   * the corpus could answer it, digested to the owner on a schedule.
   *
   * `minScore` is the BM25 score below which a question counts as unanswered.
   * BM25 is unnormalized, so this is a tuned constant rather than a
   * probability: on a corpus this size a genuinely on-topic question scores
   * several points and an off-topic one scores near zero, which leaves a wide
   * gap to sit in. Term coverage is checked alongside it (see gaps.ts) so a
   * single incidental word match cannot pass on its own.
   *
   * Set `digestDays` to 0 to keep the log without ever sending a digest.
   */
  gaps: {
    enabled: (process.env.QUERY_LOG ?? "1") !== "0",
    minScore: Number(process.env.GAP_MIN_SCORE ?? 1),
    retentionDays: Number(process.env.GAP_RETENTION_DAYS ?? 90),
    digestDays: Number(process.env.GAP_DIGEST_DAYS ?? 7),
  },

  /**
   * Repeat-caller escalation, keyed on the calling number, to blunt someone
   * dialing over and over to run up cost. Counting the calls a number has
   * placed within `windowHours`:
   *  - up to `limitedAfter`: normal, full experience.
   *  - past `limitedAfter`:  limited — the bot is told to keep it short, and
   *    the barge-in call is hard-capped at `limitedMaxSeconds`.
   *  - past `blockAfter`:    blocked — a brief spoken message, then hang up,
   *    without ever engaging the (paid) speech/LLM pipeline.
   */
  abuse: {
    windowHours: parseInt(process.env.ABUSE_WINDOW_HOURS ?? "24", 10),
    limitedAfter: parseInt(process.env.ABUSE_LIMITED_AFTER ?? "3", 10),
    blockAfter: parseInt(process.env.ABUSE_BLOCK_AFTER ?? "6", 10),
    limitedMaxSeconds: parseInt(process.env.ABUSE_LIMITED_MAX_SECONDS ?? "120", 10),
    /**
     * Hard ceiling on ANY realtime call, in seconds (default 10 minutes).
     * Every answered minute costs real money, so a stuck line, an autodialer
     * that never hangs up, or a jammed carrier leg can't run an open-ended
     * bill. Genuine conversations rarely approach this; raise it if yours do.
     */
    maxCallSeconds: parseInt(process.env.MAX_CALL_SECONDS ?? "600", 10),
    /**
     * Hang up if the caller has said NOTHING this long after the call starts
     * (default 75s: the ~25s greeting plus a generous pause). War-dialers and
     * robocall probes connect and sit silent; without this they hold a paid
     * STT stream until the duration cap. A real caller always speaks sooner —
     * and typed input from a paired browser counts as speech too. 0 disables.
     */
    silentHangupSeconds: parseInt(process.env.SILENT_HANGUP_SECONDS ?? "75", 10),
  },

  /**
   * Call recording (realtime/barge-in path). When enabled, each call's audio is
   * captured as a stereo WAV (caller left, Elle right) and uploaded to a Google
   * Cloud Storage bucket alongside a JSON sidecar (metadata + transcript). Off by
   * default: recording has consent/compliance implications — see docs/recording.md.
   * Auth is ADC (the Cloud Run service account needs write access to the bucket).
   */
  recording: {
    enabled: process.env.RECORDING_ENABLED === "true",
    bucket: process.env.RECORDING_BUCKET ?? "",
    prefix: (process.env.RECORDING_PREFIX ?? "recordings").replace(/^\/+|\/+$/g, ""),
  },

  /**
   * Intro-call booking. Elle's goal is to book callers for a short intro call
   * with the owner. When BOOKING_CALENDAR_ID is set, she creates a *tentative*
   * hold on that Google Calendar (auth via ADC — share the calendar with the
   * Cloud Run service account and grant "Make changes to events"); the owner
   * confirms it. With no calendar configured she still captures the request as a
   * lead. See docs/booking.md.
   */
  booking: {
    /**
     * Extra calendars consulted for conflicts but never written to. Elle books
     * onto the booking calendar alone; without this she also books straight
     * over anything on a personal one, because free/busy only ever saw the one
     * it writes to. Comma-separated calendar ids, each shared with the service
     * account at "See only free/busy" or better.
     */
    busyCalendars: (process.env.BOOKING_BUSY_CALENDARS ?? "")
      .split(",")
      .map((id) => id.trim())
      .filter(Boolean),
    calendarId: process.env.BOOKING_CALENDAR_ID ?? "",
    timezone: process.env.BOOKING_TIMEZONE ?? "America/Los_Angeles",
    durationMinutes: parseInt(process.env.BOOKING_DURATION_MINUTES ?? "30", 10),
    /**
     * Working hours, weekdays, in the booking timezone, as "start-end" in whole
     * hours. Only shapes the alternatives check_availability suggests when a
     * time is taken; it never stops a caller asking for, or booking, any time.
     */
    hours: (() => {
      const [a, b] = (process.env.BOOKING_HOURS ?? "9-17").split("-").map((n) => parseInt(n, 10));
      return Number.isFinite(a) && Number.isFinite(b) && a < b ? { start: a, end: b } : { start: 9, end: 17 };
    })(),
  },

  /**
   * Token that protects the owner's admin views (e.g. /admin/bookings). Empty =
   * the admin routes are disabled (return 404). Pass it as an X-Admin-Token
   * header (preferred) or ?token= query param.
   */
  admin: {
    token: secretEnv("ADMIN_TOKEN"),
  },

  /**
   * Telegram owner intake. The owner messages Elle on Telegram ("text Jane at
   * +1555... about the analytics role") and she sends the first-contact SMS and
   * tracks the thread. botToken comes from @BotFather; secret is a random string
   * you also pass to setWebhook (Telegram echoes it back in a header so nobody
   * else can post to the endpoint); ownerId is your numeric Telegram user id —
   * anyone else who finds the bot is ignored. Leave botToken empty to disable.
   */
  telegram: {
    botToken: secretEnv("TELEGRAM_BOT_TOKEN"),
    ownerId: secretEnv("TELEGRAM_OWNER_ID"),
    secret: secretEnv("TELEGRAM_WEBHOOK_SECRET"),
  },
};
