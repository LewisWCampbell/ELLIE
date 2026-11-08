/**
 * The data layer: one SQLite file (WAL mode) holding everything Elle remembers
 * across calls — callers and per-number facts (including sms_consent, the A2P
 * consent ledger), call records with full transcripts, tentative bookings,
 * owner-directed outreach threads, the anonymous query log that drives the
 * knowledge-gap report, and a small key-value settings store (e.g. the runtime
 * voice-architecture override). Schema is created idempotently at
 * import; every query below is a prepared statement. SQLite on a single Cloud
 * Run instance (min-instances=1) is deliberate: zero ops at portfolio volume,
 * with a documented path to Firestore/Cloud SQL when durability across
 * revisions matters more than simplicity.
 */
import Database from "better-sqlite3";
import { config } from "./config";

const db = new Database(config.sqlitePath);
db.pragma("journal_mode = WAL");

db.exec(`
  CREATE TABLE IF NOT EXISTS callers (
    phone_number TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS calls (
    call_sid TEXT PRIMARY KEY,
    phone_number TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT
  );

  CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number TEXT NOT NULL,
    category TEXT NOT NULL,
    fact TEXT NOT NULL,
    call_sid TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number TEXT,
    call_sid TEXT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    topic TEXT,
    start_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    event_link TEXT,
    event_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS outreach (
    phone_number TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT,
    message TEXT,
    status TEXT NOT NULL DEFAULT 'sent',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS sms_consent (
    phone_number TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    question TEXT NOT NULL,
    top_score REAL NOT NULL,
    top_source TEXT,
    top_heading TEXT,
    answered INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_facts_phone ON facts(phone_number);
  CREATE INDEX IF NOT EXISTS idx_transcripts_call ON transcripts(call_sid);
  CREATE INDEX IF NOT EXISTS idx_bookings_phone ON bookings(phone_number);
  CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);
`);

// Reminder bookkeeping. Nullable and added by migration so an existing
// database keeps working: a booking made before reminders shipped simply has
// no reminder recorded, which the due query reads as "not sent yet".
for (const [col, decl] of [
  ["reminder_sent_at", "TEXT"],
  ["confirmed_at", "TEXT"],
] as const) {
  try {
    db.exec(`ALTER TABLE bookings ADD COLUMN ${col} ${decl}`);
  } catch {
    /* already present */
  }
}

/**
 * The slot lock. Booking reads free/busy and then writes, and between those
 * two steps a second caller can pass the same check: at a handful of
 * concurrent callers two people are offered, and both take, the same time.
 * A unique index on the start time makes the database settle it — the second
 * INSERT throws instead of double-booking — which is the only place that can
 * settle it correctly, since the two callers may be on different instances.
 *
 * Cancelled bookings are excluded so a freed slot can be taken again.
 */
db.exec(`
  CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_slot
    ON bookings(start_at) WHERE status != 'cancelled';
`);

// Migration for databases created before bookings.event_id existed.
{
  const cols = db.prepare("PRAGMA table_info(bookings)").all() as Array<{ name: string }>;
  if (!cols.some((c) => c.name === "event_id")) {
    db.exec("ALTER TABLE bookings ADD COLUMN event_id TEXT");
  }
}

export interface Fact {
  id: number;
  phone_number: string;
  category: string;
  fact: string;
  call_sid: string | null;
  created_at: string;
}

const upsertCallerStmt = db.prepare(
  "INSERT INTO callers (phone_number) VALUES (?) ON CONFLICT(phone_number) DO NOTHING"
);
const insertCallStmt = db.prepare(
  "INSERT INTO calls (call_sid, phone_number) VALUES (?, ?) ON CONFLICT(call_sid) DO NOTHING"
);
const endCallStmt = db.prepare(
  "UPDATE calls SET ended_at = datetime('now') WHERE call_sid = ?"
);
const insertFactStmt = db.prepare(
  "INSERT INTO facts (phone_number, category, fact, call_sid) VALUES (?, ?, ?, ?)"
);
const factsForCallerStmt = db.prepare(
  "SELECT * FROM facts WHERE phone_number = ? ORDER BY created_at ASC"
);
const insertTranscriptStmt = db.prepare(
  "INSERT INTO transcripts (call_sid, role, content) VALUES (?, ?, ?)"
);
const countRecentCallsStmt = db.prepare(
  "SELECT COUNT(*) AS n FROM calls WHERE phone_number = ? AND started_at >= datetime('now', ?)"
);

/**
 * Called whenever anything about a call changes, so a durable copy can be made.
 *
 * A hook rather than a direct import because this file is the bottom of the
 * dependency graph: src/archive registers itself here at startup, which keeps
 * the archive optional, keeps db.ts free of network code, and avoids the
 * import cycle that a direct call would create.
 */
type CallChangedHook = (callSid: string, urgent: boolean) => void;
let callChanged: CallChangedHook = () => {};

export function setCallChangedHook(hook: CallChangedHook): void {
  callChanged = hook;
}

function noteChange(callSid: string | null | undefined, urgent = false): void {
  if (!callSid) return;
  try {
    callChanged(callSid, urgent);
  } catch {
    // Archiving must never be able to break a live call.
  }
}

export function recordCallStart(callSid: string, phoneNumber: string): void {
  upsertCallerStmt.run(phoneNumber);
  insertCallStmt.run(callSid, phoneNumber);
  // Urgent: a call that connects and drops before a word is spoken still has
  // to leave a record, and that record is written now rather than at the next
  // periodic flush, which the call may not live long enough to see.
  noteChange(callSid, true);
}

export function recordCallEnd(callSid: string): void {
  endCallStmt.run(callSid);
  noteChange(callSid, true);
}

export function saveFact(
  phoneNumber: string,
  category: string,
  fact: string,
  callSid: string | null
): Fact {
  const info = insertFactStmt.run(phoneNumber, category, fact, callSid);
  noteChange(callSid);
  return {
    id: Number(info.lastInsertRowid),
    phone_number: phoneNumber,
    category,
    fact,
    call_sid: callSid,
    created_at: new Date().toISOString(),
  };
}

export function getFactsForCaller(phoneNumber: string): Fact[] {
  return factsForCallerStmt.all(phoneNumber) as Fact[];
}

/** How many calls this number has placed within the last `windowHours` hours. */
export function countRecentCalls(phoneNumber: string, windowHours: number): number {
  const row = countRecentCallsStmt.get(phoneNumber, `-${windowHours} hours`) as { n: number };
  return row.n;
}

export function saveTranscript(callSid: string, role: "caller" | "assistant", content: string): void {
  insertTranscriptStmt.run(callSid, role, content);
  noteChange(callSid);
}

export interface TranscriptLine {
  role: string;
  content: string;
  created_at: string;
}

const transcriptsForCallStmt = db.prepare(
  "SELECT role, content, created_at FROM transcripts WHERE call_sid = ? ORDER BY id ASC"
);

/** Turn-by-turn transcript for one call, oldest first. */
export function getTranscripts(callSid: string): TranscriptLine[] {
  return transcriptsForCallStmt.all(callSid) as TranscriptLine[];
}

export interface Booking {
  phoneNumber: string | null;
  callSid: string | null;
  name: string;
  email: string;
  topic: string | null;
  startAt: string;
  status: string;
  eventLink: string | null;
  eventId?: string | null;
}

const insertBookingStmt = db.prepare(
  `INSERT INTO bookings (phone_number, call_sid, name, email, topic, start_at, status, event_link, event_id)
   VALUES (@phoneNumber, @callSid, @name, @email, @topic, @startAt, @status, @eventLink, @eventId)`
);
const bookingsForCallerStmt = db.prepare(
  "SELECT * FROM bookings WHERE phone_number = ? ORDER BY created_at DESC"
);

/** Record an intro-call booking (tentative hold or captured lead). */
export function saveBooking(b: Booking): void {
  insertBookingStmt.run({
    phoneNumber: b.phoneNumber,
    callSid: b.callSid,
    name: b.name,
    email: b.email,
    topic: b.topic,
    startAt: b.startAt,
    status: b.status,
    eventLink: b.eventLink,
    eventId: b.eventId ?? null,
  });
  // Urgent: a booking is the whole point of the call, and it is the one record
  // whose loss the caller would actually notice.
  noteChange(b.callSid, true);
}

export function getBookings(phoneNumber: string): unknown[] {
  return bookingsForCallerStmt.all(phoneNumber);
}

/** A stored booking row, id included — the shape the SMS tools work with. */
export interface StoredBooking {
  id: number;
  phone_number: string | null;
  /** The call this booking came out of, when it came from one. Null for the
   *  web chat and for SMS, which have no call to attribute it to. */
  call_sid: string | null;
  name: string;
  email: string;
  topic: string | null;
  start_at: string;
  status: string;
  event_id: string | null;
}

const bookingsByPhoneStmt = db.prepare(
  `SELECT id, phone_number, call_sid, name, email, topic, start_at, status, event_id
   FROM bookings WHERE phone_number = ? AND status != 'canceled'
   ORDER BY start_at ASC`
);
const bookingByIdStmt = db.prepare(
  `SELECT id, phone_number, call_sid, name, email, topic, start_at, status, event_id
   FROM bookings WHERE id = ?`
);
const updateBookingTimeStmt = db.prepare(
  "UPDATE bookings SET start_at = @startAt, status = 'pending' WHERE id = @id"
);
const updateBookingStatusStmt = db.prepare(
  "UPDATE bookings SET status = @status WHERE id = @id"
);

/** Active (non-canceled) bookings for a phone number, soonest first. */
export function getBookingsForPhone(phoneNumber: string): StoredBooking[] {
  return bookingsByPhoneStmt.all(phoneNumber) as StoredBooking[];
}

export function getBookingById(id: number): StoredBooking | undefined {
  return bookingByIdStmt.get(id) as StoredBooking | undefined;
}

/** Move a booking to a new start; it goes back to pending (owner re-confirms). */
export function updateBookingTime(id: number, startAt: string): void {
  updateBookingTimeStmt.run({ id, startAt });
}

export function updateBookingStatus(id: number, status: string): void {
  updateBookingStatusStmt.run({ id, status });
}

/**
 * SMS consent ledger (A2P compliance). Tracks each number's opt state and how
 * it got there ("inbound_text" = they texted us first, "sms_stop"/"sms_start" =
 * keyword, "verbal" = consent given on a call). Timestamped — this plus the
 * transcripts table is the consent record carriers/audits ask for.
 */
export interface SmsConsent {
  phone_number: string;
  status: "opted_in" | "opted_out";
  source: string | null;
  updated_at: string;
}

const setSmsConsentStmt = db.prepare(
  `INSERT INTO sms_consent (phone_number, status, source, updated_at)
   VALUES (@phoneNumber, @status, @source, datetime('now'))
   ON CONFLICT(phone_number)
   DO UPDATE SET status = @status, source = @source, updated_at = datetime('now')`
);
const getSmsConsentStmt = db.prepare("SELECT * FROM sms_consent WHERE phone_number = ?");

export function setSmsConsent(phoneNumber: string, status: "opted_in" | "opted_out", source: string): void {
  setSmsConsentStmt.run({ phoneNumber, status, source });
}

export function getSmsConsent(phoneNumber: string): SmsConsent | undefined {
  return getSmsConsentStmt.get(phoneNumber) as SmsConsent | undefined;
}

/**
 * Owner-initiated outreach ledger. One row per number Elle has texted at
 * Lewis's request (via the Telegram intake). The stored message is what she
 * sent, so the SMS agent has the full context when the person replies.
 */
export interface OutreachRow {
  phone_number: string;
  name: string;
  topic: string | null;
  message: string | null;
  status: string;
  created_at: string;
}

const upsertOutreachStmt = db.prepare(
  `INSERT INTO outreach (phone_number, name, topic, message, status, created_at)
   VALUES (@phoneNumber, @name, @topic, @message, 'sent', datetime('now'))
   ON CONFLICT(phone_number)
   DO UPDATE SET name = @name, topic = @topic, message = @message,
                 status = 'sent', created_at = datetime('now')`
);
const getOutreachStmt = db.prepare("SELECT * FROM outreach WHERE phone_number = ?");
const updateOutreachStatusStmt = db.prepare(
  "UPDATE outreach SET status = ? WHERE phone_number = ?"
);
const recentOutreachStmt = db.prepare(
  "SELECT * FROM outreach ORDER BY created_at DESC LIMIT ?"
);

export function saveOutreach(
  phoneNumber: string,
  name: string,
  topic: string | null,
  message: string | null
): void {
  upsertOutreachStmt.run({ phoneNumber, name, topic, message });
}

export function getOutreach(phoneNumber: string): OutreachRow | undefined {
  return getOutreachStmt.get(phoneNumber) as OutreachRow | undefined;
}

export function updateOutreachStatus(phoneNumber: string, status: string): void {
  updateOutreachStatusStmt.run(status, phoneNumber);
}

/** Most recent outreach rows, newest first. */
export function getRecentOutreach(limit = 10): OutreachRow[] {
  return recentOutreachStmt.all(limit) as OutreachRow[];
}

/** Tiny key-value store for owner-set runtime toggles (e.g. voice mode). */
const setSettingStmt = db.prepare(
  `INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')`
);
const getSettingStmt = db.prepare("SELECT value FROM settings WHERE key = ?");

export function setSetting(key: string, value: string): void {
  setSettingStmt.run(key, value);
}

export function getSetting(key: string): string | null {
  const row = getSettingStmt.get(key) as { value: string } | undefined;
  return row?.value ?? null;
}

export interface CallRow {
  call_sid: string;
  phone_number: string;
  started_at: string;
  ended_at: string | null;
}

const recentCallsStmt = db.prepare(
  "SELECT call_sid, phone_number, started_at, ended_at FROM calls ORDER BY started_at DESC LIMIT ?"
);

/** Most recent calls, newest first (for the owner's /admin/calls view). */
export function getRecentCalls(limit = 25): CallRow[] {
  return recentCallsStmt.all(limit) as CallRow[];
}

const callsSinceStmt = db.prepare(
  `SELECT call_sid, phone_number, started_at, ended_at
     FROM calls
    WHERE started_at >= datetime('now', ?)
    ORDER BY started_at DESC
    LIMIT ?`
);

/**
 * Calls that started within the last `hours`, newest first. Backs the owner's
 * "has anyone called?" question from the intake channel, where a time window is
 * the natural unit rather than a row count.
 */
export function getCallsSince(hours: number, limit = 20): CallRow[] {
  return callsSinceStmt.all(`-${hours} hours`, limit) as CallRow[];
}

const factsForCallStmt = db.prepare(
  `SELECT id, phone_number, category, fact, call_sid, created_at
     FROM facts WHERE call_sid = ? ORDER BY id`
);

/**
 * Facts saved during ONE call, as opposed to everything ever learned about the
 * number. Reporting on a specific call has to use this: notes from a previous
 * conversation on a shared phone are exactly what must not be attributed to
 * whoever called this time.
 */
export function getFactsForCall(callSid: string): Fact[] {
  return factsForCallStmt.all(callSid) as Fact[];
}

export interface BookingRow {
  phone_number: string | null;
  call_sid: string | null;
  name: string;
  email: string;
  topic: string | null;
  start_at: string;
  status: string;
  event_link: string | null;
  created_at: string;
}

const pendingBookingsStmt = db.prepare(
  `SELECT phone_number, call_sid, name, email, topic, start_at, status, event_link, created_at
   FROM bookings WHERE status = 'pending' ORDER BY start_at ASC`
);

/** Every booking still awaiting the owner's confirmation, soonest first. */
export function getPendingBookings(): BookingRow[] {
  return pendingBookingsStmt.all() as BookingRow[];
}

// --- durable snapshots -------------------------------------------------------
//
// Cloud Run gives every instance a fresh, empty filesystem, so this database
// exists only for the life of the container: deploy, scale to zero, or a
// routine instance replacement, and every call ever taken is gone. That is
// fine for scratch state and unacceptable for the record of who called and
// what was said.
//
// A snapshot is everything about ONE call, in one self-describing object. Per
// call rather than per table because a call is the unit that is written once
// and never changed afterwards, which makes each snapshot immutable, safely
// re-uploadable by whichever instance happens to be handling that call, and
// replayable in any order.

export interface CallSnapshot {
  /** Schema version, so an old archive stays readable after a shape change. */
  v: 1;
  call: CallRow;
  transcript: TranscriptLine[];
  facts: Fact[];
  bookings: BookingRow[];
}

const callByIdStmt = db.prepare(
  "SELECT call_sid, phone_number, started_at, ended_at FROM calls WHERE call_sid = ?"
);
const bookingsForCallStmt = db.prepare(
  `SELECT phone_number, call_sid, name, email, topic, start_at, status, event_link, created_at
     FROM bookings WHERE call_sid = ? ORDER BY created_at ASC`
);

/** Everything worth keeping about one call, or null if there is no such call. */
export function snapshotCall(callSid: string): CallSnapshot | null {
  const call = callByIdStmt.get(callSid) as CallRow | undefined;
  if (!call) return null;
  return {
    v: 1,
    call,
    transcript: getTranscripts(callSid),
    facts: getFactsForCall(callSid),
    bookings: bookingsForCallStmt.all(callSid) as BookingRow[],
  };
}

const hasCallStmt = db.prepare("SELECT 1 FROM calls WHERE call_sid = ?");

export function hasCall(callSid: string): boolean {
  return hasCallStmt.get(callSid) !== undefined;
}

const restoreCallerStmt = db.prepare(
  "INSERT INTO callers (phone_number, first_seen_at) VALUES (?, ?) ON CONFLICT(phone_number) DO NOTHING"
);
const restoreCallStmt = db.prepare(
  `INSERT INTO calls (call_sid, phone_number, started_at, ended_at) VALUES (?, ?, ?, ?)
   ON CONFLICT(call_sid) DO NOTHING`
);
const restoreTranscriptStmt = db.prepare(
  "INSERT INTO transcripts (call_sid, role, content, created_at) VALUES (?, ?, ?, ?)"
);
const restoreFactStmt = db.prepare(
  "INSERT INTO facts (phone_number, category, fact, call_sid, created_at) VALUES (?, ?, ?, ?, ?)"
);
const restoreBookingStmt = db.prepare(
  `INSERT INTO bookings (phone_number, call_sid, name, email, topic, start_at, status, event_link, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
);

/**
 * Replay one archived call into this instance's database.
 *
 * Returns false when the call is already here, and that check is what makes
 * restoring idempotent: transcript and fact rows carry autoincrement ids, so
 * replaying a call twice would duplicate every line of the conversation. The
 * call row is the natural key, and it is written first on the live path, so
 * its presence is a reliable "already have this one".
 */
export const restoreCall = db.transaction((snap: CallSnapshot): boolean => {
  if (!snap?.call?.call_sid) return false;
  if (hasCall(snap.call.call_sid)) return false;

  restoreCallerStmt.run(snap.call.phone_number, snap.call.started_at);
  restoreCallStmt.run(
    snap.call.call_sid,
    snap.call.phone_number,
    snap.call.started_at,
    snap.call.ended_at
  );
  for (const line of snap.transcript ?? []) {
    restoreTranscriptStmt.run(snap.call.call_sid, line.role, line.content, line.created_at);
  }
  for (const f of snap.facts ?? []) {
    restoreFactStmt.run(f.phone_number, f.category, f.fact, f.call_sid, f.created_at);
  }
  for (const b of snap.bookings ?? []) {
    restoreBookingStmt.run(
      b.phone_number,
      b.call_sid,
      b.name,
      b.email,
      b.topic,
      b.start_at,
      b.status,
      b.event_link,
      b.created_at
    );
  }
  return true;
}) as (snap: CallSnapshot) => boolean;

// --- caller-visible fact corrections ----------------------------------------
//
// Both are scoped by phone number as well as id: the id alone would let one
// paired session edit another caller's notes by guessing small integers, and
// nothing about a row id is secret. The phone number comes from the session,
// never from the browser.

const updateFactStmt = db.prepare(
  "UPDATE facts SET fact = ? WHERE id = ? AND phone_number = ?"
);
const deleteFactStmt = db.prepare(
  "DELETE FROM facts WHERE id = ? AND phone_number = ?"
);

/** Rewrite one fact the caller corrected on the paired page. */
export function updateFact(id: number, phoneNumber: string, fact: string): boolean {
  return updateFactStmt.run(fact, id, phoneNumber).changes > 0;
}

/** Remove one fact the caller asked to forget. */
export function deleteFact(id: number, phoneNumber: string): boolean {
  return deleteFactStmt.run(id, phoneNumber).changes > 0;
}

// --- reminders ---------------------------------------------------------------

export interface DueBooking {
  id: number;
  phone_number: string | null;
  name: string;
  email: string;
  start_at: string;
  status: string;
}

/**
 * Bookings starting inside the window that have not been reminded yet.
 *
 * The lower bound matters as much as the upper one: without it, a job that has
 * not run for a day would "remind" people about appointments that already
 * happened. Cancelled bookings are excluded, and reminder_sent_at is the idempotence
 * key, so running the job twice in one window sends one message.
 */
const dueRemindersStmt = db.prepare(
  `SELECT id, phone_number, name, email, start_at, status
     FROM bookings
    WHERE reminder_sent_at IS NULL
      AND status != 'cancelled'
      AND start_at > ?
      AND start_at <= ?
    ORDER BY start_at ASC
    LIMIT 50`
);

export function getBookingsDueForReminder(nowISO: string, untilISO: string): DueBooking[] {
  return dueRemindersStmt.all(nowISO, untilISO) as DueBooking[];
}

const markReminderStmt = db.prepare(
  "UPDATE bookings SET reminder_sent_at = ? WHERE id = ? AND reminder_sent_at IS NULL"
);

/**
 * Claim a reminder. Returns false if another instance claimed it first, which
 * is what stops two Cloud Run containers from both texting the same person:
 * the UPDATE is the claim, and only one of them changes a row.
 */
export function claimReminder(id: number, atISO = new Date().toISOString()): boolean {
  return markReminderStmt.run(atISO, id).changes > 0;
}

const bookingsForCallStmt2 = db.prepare(
  "SELECT id FROM bookings WHERE call_sid = ?"
);

/** Did this call produce a booking? Used to avoid double-notifying the owner. */
export function getBookingsForCall(callSid: string): Array<{ id: number }> {
  return bookingsForCallStmt2.all(callSid) as Array<{ id: number }>;
}

/* ---------------------------------------------------------------------------
 * The query log.
 *
 * Every other table here is about one person: their number, their facts, their
 * booking. This one is deliberately not. It records what gets asked and
 * whether the knowledge base could answer it, with no phone number and no
 * call_sid, so it cannot be joined back to a caller. That is not only a
 * privacy position, it is what the table is for: the useful signal is the
 * distribution of questions across everyone, not who asked which one.
 * ------------------------------------------------------------------------ */

export interface QueryRow {
  id: number;
  channel: string;
  question: string;
  top_score: number;
  top_source: string | null;
  top_heading: string | null;
  answered: number;
  created_at: string;
}

const insertQueryStmt = db.prepare(
  `INSERT INTO queries (channel, question, top_score, top_source, top_heading, answered)
   VALUES (?, ?, ?, ?, ?, ?)`
);

export function logQuery(q: {
  channel: string;
  question: string;
  topScore: number;
  topSource?: string | null;
  topHeading?: string | null;
  answered: boolean;
}): void {
  insertQueryStmt.run(
    q.channel,
    q.question,
    q.topScore,
    q.topSource ?? null,
    q.topHeading ?? null,
    q.answered ? 1 : 0
  );
}

const queriesSinceStmt = db.prepare(
  `SELECT * FROM queries WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?`
);

export function getQueriesSince(sinceISO: string, limit = 2000): QueryRow[] {
  return queriesSinceStmt.all(sqliteTime(sinceISO), limit) as QueryRow[];
}

const pruneQueriesStmt = db.prepare("DELETE FROM queries WHERE created_at < ?");

/** Drop log entries past the retention window. Returns rows removed. */
export function pruneQueries(beforeISO: string): number {
  return pruneQueriesStmt.run(sqliteTime(beforeISO)).changes;
}

/**
 * created_at is written by SQLite's datetime('now'), which is "YYYY-MM-DD
 * HH:MM:SS" in UTC. Comparing that against a JavaScript ISO string is a string
 * comparison, and "2026-09-03T00:00:00.000Z" sorts after "2026-09-03 00:00:00"
 * on the T, so an unconverted bound silently excludes the whole day.
 */
function sqliteTime(iso: string): string {
  return iso.replace("T", " ").replace(/\.\d+Z?$/, "").replace(/Z$/, "");
}
