# Owner intake channel: setup and troubleshooting

The private Telegram channel where the owner directs Elle (send outreach, check
threads, switch voice architecture). It is owner-only and never appears in any
visitor-facing surface.

Two gates protect the webhook, and **both fail silently by design** (an attacker
should learn nothing). That makes misconfiguration invisible unless you read the
logs, so this runbook exists.

## Required configuration

| Variable | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather. **~46 chars**, shaped `1234567890:AAH...` |
| `TELEGRAM_WEBHOOK_SECRET` | Random string, registered with `setWebhook` **and** stored as the secret |
| `TELEGRAM_OWNER_ID` | Your **numeric** user id, not your `@username` |

## First-time setup

Store the bot token (visible `read` so you can confirm what landed):

```bash
read -r TG_TOKEN
echo "length: ${#TG_TOKEN}"        # must be ~46
curl -s "https://api.telegram.org/bot${TG_TOKEN}/getMe"; echo   # must be "ok":true
printf '%s' "$TG_TOKEN" | gcloud secrets versions add telegram-bot-token --data-file=-
```

Mint and register the webhook secret. Both sides come from one variable, so they
cannot drift:

```bash
NEW=$(openssl rand -hex 32)
printf '%s' "$NEW" | gcloud secrets versions add telegram-webhook-secret --data-file=-
curl -s "https://api.telegram.org/bot${TG_TOKEN}/setWebhook" \
  -d "url=https://talk.lewiswcampbell.com/telegram" \
  -d "secret_token=${NEW}" -d "drop_pending_updates=true"; echo
unset TG_TOKEN NEW
```

Deploy (Cloud Run reads secrets at instance start, so a new version does nothing
without a new revision), then message the bot. With no owner id set it replies
with your numeric id; set it and redeploy:

```bash
gcloud run deploy letstalk-elle --source . --region us-central1
gcloud run services update letstalk-elle --region us-central1 \
  --update-env-vars TELEGRAM_OWNER_ID=<the number it told you>
```

## Diagnosing silence

Telegram never reports a rejected update as an error, so the service logs are the
only witness:

```bash
gcloud run services logs read letstalk-elle --region us-central1 --limit 100 | grep -i telegram
```

| Log line | Meaning | Fix |
|---|---|---|
| *(no telegram lines at all)* | Updates are not arriving | Re-run `setWebhook`; check `getWebhookInfo` |
| `dropped update with bad secret token` | Webhook secret ≠ stored secret | Re-mint both from one variable (above), then redeploy |
| `ignored message from non-owner N` | Delivery works; wrong owner id | Set `TELEGRAM_OWNER_ID=N` |
| `lewis: "..."` then `sendMessage 401/404` | Inbound fine, token bad | Re-store the bot token |

`/admin/status` shows the same chain (registered URL vs this service, pending
updates, Telegram's own `last_error`) without shell access.

## Traps that cost real time

- **Truncated bot token.** A partial copy stores fine and fails forever. Always
  check the length is ~46 and that `getMe` returns `"ok":true` *before* storing.
- **Secret drift.** Registering the webhook in one session and storing the secret
  in another is how the values diverge. Always set both from the same variable.
- **Trailing newline.** `echo` appends `\n`; Secret Manager stores it; Cloud Run
  injects it verbatim; every comparison fails. Always use `printf '%s'`.
  `config.ts` now trims credentials defensively, but store them clean anyway.
- **Stale revision.** `:latest` secret references resolve at instance startup.
  Adding a version without deploying changes nothing.
- **Paste hazards.** A `read` command followed by a blank line in a pasted block
  consumes the blank line as its input. Run `read` on its own. Likewise, a pasted
  `set -euo pipefail` stays active in an interactive shell and closes it on the
  next failure (`set +e +u +o pipefail` to clear).
