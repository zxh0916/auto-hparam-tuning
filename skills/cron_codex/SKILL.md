---
name: cron_codex
description: Schedule a payload to be sent later to a specific Codex session or the current thread via host cron, and optionally forward the matching final answer to the user's terminal.
---

# cron_codex

Use this skill when a user wants Codex to send a message later instead of immediately.

Preferred path: use `scripts/schedule_codex_message.py`. It now writes a one-shot host `crontab` entry, launches Codex with an explicit Node 24 runtime, and sends the payload back into the target thread so the reply is produced in that same thread.

## Directory discipline

- Treat this skill directory as implementation-only. Reuse the existing files here and do not create extra helper scripts, notes, caches, logs, screenshots, test outputs, or ad-hoc artifacts in this folder unless the user explicitly asks for a code change to the skill itself.
- Do not casually add new files beside `SKILL.md` and the existing `scripts/` contents when merely using the skill.
- If temporary data is unavoidable while using the skill, write it outside this directory and clean it up after use.
- If runtime artifacts such as `__pycache__` or `.cron_codex_jobs/` are created here accidentally, remove them before finishing.
- Only edit files in this directory when the user explicitly asks to modify the skill.

## Inputs to collect

- Target session: a session id, thread name, `--last`, or the current `CODEX_THREAD_ID`
- Target time: a concrete local timestamp such as `2026-03-22 21:30`, or a relative delay
- Payload: the exact text to send
- Whether the final answer should also be forwarded to terminal
- If the user expects a specific terminal output, phrase the scheduled prompt explicitly, for example `[CRON_CODEX] 请只回复 smshl` instead of bare `smshl`

If the user gives a relative time, convert it to an absolute timestamp before scheduling.

## Workflow

1. Confirm the session target and exact scheduled time or delay.
2. If terminal forwarding is required, prefer `--session` instead of `--last`.
3. Run the scheduler script.
4. If you notice stale one-shot `cron_codex` entries that should already have fired but are still present, remove those stale entries as a follow-up cleanup and tell the user you did it.
5. Report the created job id, scheduled time, log paths, and any cleanup you performed back to the user.

## Commands

Basic schedule:

```bash
python3 scripts/schedule_codex_message.py \
  --session "<SESSION_ID>" \
  --at "2026-03-22 21:30:00" \
  --payload "your payload"
```

Schedule relative to now into the current thread:

```bash
CODEX_THREAD_ID="<CURRENT_THREAD_ID>" \
python3 scripts/schedule_codex_message.py \
  --after-seconds 300 \
  --payload "your payload"
```

Schedule and forward the matching final answer to the current user's terminals:

```bash
python3 scripts/schedule_codex_message.py \
  --session "<SESSION_ID>" \
  --at "2026-03-22 21:30:00" \
  --payload "your payload" \
  --forward-all-user-ttys \
  --bell
```

Use `--last` instead of `--session` only when terminal forwarding is not needed.

## Notes

- The script writes a one-shot `crontab` entry and returns immediately.
- Logs are written under `.cron_codex_jobs/` next to the caller's current working directory.
- The generated cron runner removes its own crontab entry after firing.
- If a one-shot entry is clearly stale and did not remove itself, it is appropriate to clean it up manually when you are already touching `crontab` for the user. Mention that cleanup explicitly in your response.
- The cron runner uses the explicit Node 24 + Codex JS path so it does not depend on cron's default PATH.
- When forwarding is enabled, the scheduler starts a one-shot watcher that forwards only the matching final answer for that exact scheduled message. By default it can broadcast to all writable terminals of the current user for portability.
- The script rejects timestamps in the past.
- For verification without sending, use `--dry-run`.
