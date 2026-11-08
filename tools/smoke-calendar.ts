/**
 * Live end-to-end check of the booking calendar. Reads free/busy, creates a
 * real tentative hold, reads it back, moves it, then deletes it.
 *
 * The offline suite (npm run verify:booking) fakes the calendar, so it proves
 * the ORCHESTRATION is right and can say nothing about whether this deployment
 * can actually reach the calendar. The things that break in production are all
 * on the other side of that fake: the API not enabled, the calendar not shared
 * with the service account, shared read-only rather than "make changes", the
 * wrong calendar id. Each of those fails at a different step, so this walks the
 * steps in order and names which one broke.
 *
 * Nothing is left behind: the hold it creates is deleted before it exits, and
 * it is created far enough out that it cannot collide with a real meeting.
 * The caller is never emailed, because every write uses sendUpdates=none.
 *
 * Run via tools/smoke-calendar.sh.
 */
import { execFileSync } from "node:child_process";

/**
 * NOTE the dynamic imports further down. src/config reads the environment once,
 * at module load, so anything this script discovers about the deployed
 * configuration has to be put into process.env BEFORE config is imported.
 * A static import at the top would freeze the config before the lookup ran.
 */

const SERVICE = process.env.SERVICE ?? "letstalk-elle";
const REGION = process.env.REGION ?? "us-central1";

interface DeployedConfig {
  calendarId?: string;
  timezone?: string;
  serviceAccount?: string;
}

/**
 * Read the relevant settings off the DEPLOYED service.
 *
 * A local checkout normally has no .env, so reading only the local environment
 * answers a question nobody asked: what this machine thinks, rather than what
 * production is actually configured with. Returns null when gcloud is absent or
 * cannot describe the service, which is a fine outcome; the caller just falls
 * back to whatever is local.
 */
function readDeployedConfig(): DeployedConfig | null {
  let raw: string;
  try {
    raw = execFileSync(
      "gcloud",
      ["run", "services", "describe", SERVICE, "--region", REGION, "--format=json"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }
    );
  } catch {
    return null;
  }
  try {
    const d = JSON.parse(raw) as {
      spec?: { template?: { spec?: { containers?: Array<{ env?: Array<{ name?: string; value?: string }> }>; serviceAccountName?: string } } };
    };
    const spec = d.spec?.template?.spec;
    const env: Record<string, string> = {};
    for (const c of spec?.containers ?? []) {
      for (const e of c.env ?? []) {
        // Secret-backed entries carry no `value`; skip them rather than
        // recording undefined.
        if (e.name && typeof e.value === "string") env[e.name] = e.value;
      }
    }
    return {
      calendarId: env.BOOKING_CALENDAR_ID,
      timezone: env.BOOKING_TIMEZONE,
      serviceAccount: spec?.serviceAccountName,
    };
  } catch {
    return null;
  }
}


const PROBE_DAYS_OUT = 180;

/**
 * A guest is attached to the probe because every real booking attaches one, and
 * that is precisely the operation Google refuses for a service account on a
 * calendar that is merely shared with it. The address is a reserved example
 * domain, so nothing can ever be delivered to it, and sendUpdates=none means
 * nothing is sent anyway.
 */
const PROBE_GUEST = "elle-smoke-test@example.com";

function fmt(d: Date, timeZone: string): string {
  try {
    return d.toLocaleString("en-US", {
      timeZone,
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return d.toISOString();
  }
}

function fail(step: string, err: unknown, hints: string[]): never {
  const message = err instanceof Error ? err.message : String(err);
  console.error(`\nFAILED at: ${step}`);
  console.error(`  ${message}\n`);
  console.error("Most likely causes, in order:");
  for (const h of hints) console.error(`  - ${h}`);
  process.exit(1);
}

async function main(): Promise<void> {
  // Resolve the environment BEFORE importing config, which reads it once at
  // module load. These are inside main() rather than at the top of the file
  // because top-level await is not available in this CommonJS build.
  let source = "local environment";
  let deployed: DeployedConfig | null = null;

  if (!process.env.BOOKING_CALENDAR_ID) {
    deployed = readDeployedConfig();
    if (deployed?.calendarId) {
      process.env.BOOKING_CALENDAR_ID = deployed.calendarId;
      if (deployed.timezone) process.env.BOOKING_TIMEZONE = deployed.timezone;
      source = `deployed service ${SERVICE}`;
    }
  }

  const { config } = await import("../src/config");
  const { createCalendarClient } = await import("../src/booking/calendar");
  const { parseStart } = await import("../src/booking/booking");

  console.log(`calendar  ${config.booking.calendarId || "(not set)"}   [from ${source}]`);
  console.log(`timezone  ${config.booking.timezone}`);
  console.log(`duration  ${config.booking.durationMinutes} min\n`);

  const cal = createCalendarClient();
  if (!cal) {
    console.error("No calendar is configured, locally or on the deployed service.");
    console.error("Booking currently captures leads and creates no calendar holds.\n");
    console.error("To turn it on:\n");
    console.error("  1. Google Calendar > your calendar > Settings and sharing >");
    console.error("     Integrate calendar > copy the Calendar ID.");
    console.error("     For your own primary calendar that is just your email address.\n");
    console.error("  2. On that same screen, under 'Share with specific people', add:");
    console.error(`       ${deployed?.serviceAccount ?? "(the account the service runs as)"}`);
    console.error("     with permission: Make changes to events");
    console.error("     Read-only is the usual mistake, and it fails only at the moment");
    console.error("     a caller tries to book.\n");
    console.error(`  3. gcloud run services update ${SERVICE} --region ${REGION} \\`);
    console.error("       --update-env-vars BOOKING_CALENDAR_ID=THE-ID-FROM-STEP-1\n");
    console.error("  4. rerun this check");
    if (!deployed) {
      console.error("\nNote: the deployed service could not be read (gcloud missing, not");
      console.error("authenticated, or no such service), so only the local environment was");
      console.error("checked. Production may well be configured.");
    }
    process.exit(2);
  }

  // The check itself is shared with /admin/calendar-check so both surfaces
  // behave identically; only the identity differs, which is the whole point.
  const { runCalendarCheck } = await import("../src/booking/calendarCheck");
  const result = await runCalendarCheck(cal, config.booking.timezone, config.booking.durationMinutes);

  for (const step of result.steps) {
    console.log(`  ${step.status.toUpperCase().padEnd(4)}  ${step.name.padEnd(28)} ${step.detail}`);
  }
  console.log("");

  if (result.diagnosis) {
    console.log(result.diagnosis);
    console.log("");
  }

  if (!result.ok) {
    console.error("Booking calendar is NOT working for these credentials.\n");
    console.error("Before changing anything: this ran as YOU, not as the account Elle books");
    console.error("with. Trying to grant your own account the Calendar scope through gcloud");
    console.error("is a dead end, because Google refuses sensitive scopes to the shared");
    console.error("gcloud OAuth client. Check the identity that matters instead:\n");
    console.error("  /admin/calendar-check   runs this same check inside the service,");
    console.error("                          as the account that actually books");
    process.exit(1);
  }

  console.log("Booking calendar is working for these credentials.");
  console.log("Note this ran as you. The authoritative check is /admin/calendar-check,");
  console.log("which runs the same steps inside the service, as the account Elle books with.");
}

void main();
