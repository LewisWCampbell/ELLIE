/**
 * Durable call archive on Cloud Storage.
 *
 * The problem this solves: Cloud Run containers are disposable. The SQLite
 * file every call is written to lives on the container's own disk, so a
 * deploy, a scale-to-zero, or a routine instance replacement takes the entire
 * call history with it. Nothing errors when that happens, which is the worst
 * part: the owner's call list simply comes back empty and reads as "nobody
 * called".
 *
 * The shape of the fix matters. The obvious move is to put the SQLite file
 * itself on shared storage, and it does not work: SQLite's locking is built on
 * POSIX file locks that object storage does not implement, so two instances
 * writing one file corrupt it. Instead each CALL is archived as its own
 * immutable object, and startup replays them back into the local database.
 * Per-call objects have no write contention by construction, because exactly
 * one instance is handling any given call, and replay is idempotent because
 * the call id is the object name.
 *
 * The local database stays exactly as it was. Every existing read path, every
 * query, every admin view is untouched; this only adds a copy going out and a
 * replay coming in.
 */
import { GoogleAuth } from "google-auth-library";
import { config } from "./config";
import * as db from "./db";

const BASE = "https://storage.googleapis.com/storage/v1/b";
const UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b";

/**
 * How long a call can sit unarchived. Short enough that an instance dying
 * mid-conversation loses a sentence or two rather than the call, long enough
 * that a talkative caller does not cause an upload per word.
 */
const FLUSH_MS = 15_000;

/** Give up re-uploading one call after this many consecutive failures. */
const MAX_ATTEMPTS = 5;

let auth: GoogleAuth | null = null;
let tokenSource: () => Promise<string> = async () => {
  auth ??= new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/devstorage.read_write"] });
  const client = await auth.getClient();
  const token = await client.getAccessToken();
  if (!token.token) throw new Error("no access token for Cloud Storage");
  return token.token;
};

/** Test seam: replaces the credential source. */
export function setTokenSourceForTests(fn: () => Promise<string>): void {
  tokenSource = fn;
}

const dirty = new Map<string, number>(); // callSid -> consecutive failures
const inFlight = new Set<string>();
let timer: NodeJS.Timeout | null = null;
let restored = 0;
let lastError = "";
let restoreNote = "not run";

export function archiveEnabled(): boolean {
  return Boolean(config.callArchiveBucket);
}

/**
 * Object names begin with the call's start time so that a lexicographic
 * listing is also a chronological one. That is what lets startup ask Cloud
 * Storage for "everything since this date" with a prefix instead of listing
 * the whole bucket and sorting it here.
 */
export function objectName(startedAt: string, callSid: string): string {
  const stamp = (startedAt || "").replace(" ", "T").replace(/:/g, "-").slice(0, 19);
  const safe = callSid.replace(/[^A-Za-z0-9._-]/g, "-");
  return `calls/${stamp || "0000-00-00T00-00-00"}_${safe}.json`;
}

async function gcs(url: string, init: RequestInit = {}): Promise<Response> {
  const token = await tokenSource();
  return fetch(url, {
    ...init,
    headers: { ...(init.headers ?? {}), Authorization: `Bearer ${token}` },
  });
}

/** Write one call's snapshot to the bucket, overwriting any earlier copy. */
async function upload(callSid: string): Promise<void> {
  const snap = db.snapshotCall(callSid);
  if (!snap) return;
  const name = objectName(snap.call.started_at, callSid);
  const res = await gcs(
    `${UPLOAD}/${encodeURIComponent(config.callArchiveBucket)}/o?uploadType=media&name=${encodeURIComponent(name)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(snap),
    }
  );
  if (!res.ok) {
    throw new Error(`archive upload ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
}

async function flushOne(callSid: string): Promise<void> {
  if (inFlight.has(callSid)) return;
  inFlight.add(callSid);
  try {
    await upload(callSid);
    dirty.delete(callSid);
    lastError = "";
  } catch (e) {
    const attempts = (dirty.get(callSid) ?? 0) + 1;
    lastError = e instanceof Error ? e.message : String(e);
    if (attempts >= MAX_ATTEMPTS) {
      // Stop retrying this one, but say so loudly. A silently abandoned call
      // is the exact failure this module exists to prevent.
      console.error(`[archive] giving up on ${callSid} after ${attempts} attempts: ${lastError}`);
      dirty.delete(callSid);
    } else {
      dirty.set(callSid, attempts);
      console.warn(`[archive] ${callSid} attempt ${attempts}: ${lastError}`);
    }
  } finally {
    inFlight.delete(callSid);
  }
}

/** Write every call that has unsaved changes. Safe to call at any time. */
export async function flushAll(): Promise<void> {
  if (!archiveEnabled()) return;
  await Promise.all([...dirty.keys()].map(flushOne));
}

/**
 * Start archiving. Registers the database hook and the periodic flush.
 *
 * Nothing here can throw into a call: the hook only marks a call dirty, and
 * every upload failure is caught. A broken archive must degrade to the old
 * ephemeral behaviour, never to a dropped call.
 */
export function startArchive(): void {
  if (!archiveEnabled()) {
    console.warn(
      "[archive] CALL_ARCHIVE_BUCKET is not set: call history lives on this " +
        "container's disk and will be lost on the next deploy."
    );
    return;
  }
  db.setCallChangedHook((callSid, urgent) => {
    if (!dirty.has(callSid)) dirty.set(callSid, 0);
    if (urgent) void flushOne(callSid);
  });
  timer ??= setInterval(() => {
    void flushAll();
  }, FLUSH_MS);
  timer.unref?.();
  console.log(`[archive] on, bucket gs://${config.callArchiveBucket}`);
}

export function stopArchive(): void {
  if (timer) clearInterval(timer);
  timer = null;
}

interface ListedObject {
  name: string;
}

/** One page of object names at or after `startOffset`. */
async function listPage(startOffset: string, pageToken?: string): Promise<{ items: ListedObject[]; next?: string }> {
  const params = new URLSearchParams({
    prefix: "calls/",
    startOffset,
    maxResults: "1000",
    fields: "items(name),nextPageToken",
  });
  if (pageToken) params.set("pageToken", pageToken);
  const res = await gcs(`${BASE}/${encodeURIComponent(config.callArchiveBucket)}/o?${params}`);
  if (!res.ok) {
    throw new Error(`archive list ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data = (await res.json()) as { items?: ListedObject[]; nextPageToken?: string };
  return { items: data.items ?? [], next: data.nextPageToken };
}

async function download(name: string): Promise<db.CallSnapshot | null> {
  const res = await gcs(
    `${BASE}/${encodeURIComponent(config.callArchiveBucket)}/o/${encodeURIComponent(name)}?alt=media`
  );
  if (!res.ok) return null;
  try {
    return (await res.json()) as db.CallSnapshot;
  } catch {
    return null;
  }
}

/**
 * Replay archived calls into this instance's database.
 *
 * Runs after the server is already listening, because history is not needed to
 * answer a phone call and a slow bucket must not delay accepting one. Calls
 * already present are skipped, so this is safe to run more than once.
 */
export async function restoreFromArchive(days = config.callArchiveRestoreDays): Promise<number> {
  if (!archiveEnabled()) {
    restoreNote = "archive off";
    return 0;
  }
  const cutoff = new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10);
  const startOffset = `calls/${cutoff}`;

  let names: string[] = [];
  try {
    let page = await listPage(startOffset);
    names = page.items.map((i) => i.name);
    while (page.next) {
      page = await listPage(startOffset, page.next);
      names.push(...page.items.map((i) => i.name));
    }
  } catch (e) {
    lastError = e instanceof Error ? e.message : String(e);
    restoreNote = `list failed: ${lastError}`;
    console.error(`[archive] ${restoreNote}`);
    return 0;
  }

  let count = 0;
  for (const name of names) {
    const snap = await download(name);
    if (!snap) continue;
    try {
      if (db.restoreCall(snap)) count++;
    } catch (e) {
      console.warn(`[archive] could not replay ${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  restored = count;
  restoreNote = `${count} of ${names.length} archived call(s) replayed, last ${days} days`;
  console.log(`[archive] ${restoreNote}`);
  return count;
}

export function archiveStatus(): {
  enabled: boolean;
  bucket: string;
  restored: number;
  pending: number;
  note: string;
  lastError: string;
} {
  return {
    enabled: archiveEnabled(),
    bucket: config.callArchiveBucket,
    restored,
    pending: dirty.size,
    note: restoreNote,
    lastError,
  };
}
