# Intro-call booking

Elle's job is to show off the owner's skills and then book the caller for a short
intro call. When the caller agrees, she calls the `book_intro_call` tool, which
creates a **tentative** hold on the owner's Google Calendar and tells the caller
the time is pending the owner's confirmation. The owner reviews the tentative
event and confirms (and invites the caller) when it works.

If no calendar is configured, the tool still records the request as a lead in the
`bookings` table, so the goal degrades gracefully.

## Flow

```
caller asks about a time → check_availability(start)          (read, no side effects)
   → FREE: …   | BUSY: … Nearby open times: …   | UNKNOWN: …
caller picks a time → Elle collects name + email, reads it back, gets a yes
   → book_intro_call(name, email, start, topic)                (write)
       → freeBusy check on the owner's calendar
       → creates a TENTATIVE event (status: "tentative")
       → saves a pending row in `bookings`
   → Elle: "You're penciled in for <time>, pending <owner>'s confirmation.
            He'll confirm by email."
```

- **Read-back first.** Elle is instructed to read the exact date, time, and
  timezone back to the caller and get a clear yes before calling the tool, so the
  proposed slot is confirmed in plain language before anything is written.
- **Tentative, not confirmed.** Events are created with `status: "tentative"`. The
  caller is attached as a **guest**, but `sendUpdates=none` means they are **not**
  emailed yet. So confirming is one action: open the tentative event and hit Send,
  which flips it and invites the caller. Elle never implies the time is final.
- **Availability is its own tool.** `check_availability(start)` answers "is that
  time free?" before anything is collected or written. It makes one `freeBusy`
  query covering the next three days, so when the time is taken it can offer up
  to three nearby open slots within working hours (`BOOKING_HOURS`, default
  `9-17`, weekdays). Working hours only shape those suggestions; a caller can
  still ask for any time. Every result starts with `FREE:`, `BUSY:` or `UNKNOWN:`,
  and a calendar that cannot be read is always `UNKNOWN:`, never `FREE:`. Elle is
  told never to say a time is free or taken until the result comes back.
- **The write re-checks.** `book_intro_call` runs its own `freeBusy` check at
  write time, because someone can take the slot between the read and the write;
  the unique slot index catches the last few milliseconds of that race.
- **All channels.** Phone (realtime + cascaded) and the website chat all expose
  the tool. On the web the visitor is anonymous, so the email is the contact.

## What's built

- `src/booking/calendar.ts` — `CalendarClient` (freeBusy + create tentative) over
  the Calendar REST API, authed by ADC. Returns null when no calendar is set.
- `src/booking/availability.ts` — the `check_availability` read: busy ranges,
  nearby openings in working hours, results phrased the way Elle says them.
- `src/booking/booking.ts` — the `book_intro_call` tool schema + shared
  `handleBookIntroCall` handler (validation, availability, tentative hold, DB
  save). The calendar client is injectable for tests.
- Both tools are wired into `ai.ts`, `webchat.ts`, `sms.ts` and `twilioBridge.ts`; persona retuned so
  booking is Elle's goal (`persona.ts`), with the current time supplied so the
  model can compute an ISO start.
- `npm run verify:booking` — offline test (no network): tentative-on-free,
  refuse-on-busy, reject bad input, capture-without-calendar.

## Setup (Google Calendar)

Auth is ADC — the same Cloud Run service account that reaches Vertex. No OAuth.

1. Enable the API: `gcloud services enable calendar-json.googleapis.com`.
2. **Share the calendar with the service account.** In Google Calendar settings
   for the calendar you want holds on → *Share with specific people* → add
   `${PROJNUM}-compute@developer.gserviceaccount.com` with **"Make changes to
   events"**.
3. Set the env and redeploy:
   ```bash
   gcloud run services update letstalk-elle --region us-central1 \
     --update-env-vars BOOKING_CALENDAR_ID=you@gmail.com,BOOKING_TIMEZONE=America/Los_Angeles,BOOKING_DURATION_MINUTES=30
   ```
   `BOOKING_CALENDAR_ID` is usually your calendar's address (often your email), or
   a dedicated calendar's id from its settings.

## Seeing what's pending

Set `ADMIN_TOKEN` and open **`/admin/bookings`** (pass the token as an
`X-Admin-Token` header or `?token=` query param). It lists every tentative hold
awaiting your confirmation, soonest first, each with a "Confirm in Calendar" link
straight to the event. `/admin/bookings.json` returns the same as JSON. With no
`ADMIN_TOKEN` set the routes 404, so caller data is never exposed by default.

```bash
gcloud run services update letstalk-elle --region us-central1 \
  --update-env-vars ADMIN_TOKEN=some-long-random-string
# then: https://talk.lewiswcampbell.com/admin/bookings  (with the header/param)
```

This is a lightweight token guard for a demo; put real auth in front for
production, and prefer the header over the query param (URLs get logged).

## Notes / limits

- **Time parsing** is the model's job: it converts "next Tuesday at 2" to an ISO
  start using the current time supplied in the prompt. Because you confirm every
  booking, small errors are caught before anything is final; for stricter parsing
  add explicit read-backs ("so that's Tuesday the 25th at 2pm Pacific?").
- **Reminders/confirmation UI** for the owner are out of scope here: tentative
  events simply appear on the calendar. A follow-up could email you the pending
  list or add a one-click confirm.
