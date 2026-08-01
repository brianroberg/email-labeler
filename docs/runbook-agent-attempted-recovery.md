# Runbook: recover threads dropped to `agent/attempted` (issue #64)

Owner-run, manual. Nothing in this runbook is automated by the daemon, and no
step should be run by an agent with Gmail write access — the whole point is a
human deciding which threads re-enter the queue.

## Why these threads are lost

Under Ollama, the local tier's thinking-disable was silently inert (issue
#64): hard person threads spent the entire `max_tokens` budget in the native
reasoning channel, returned empty content (`LLMContentError`), took
`FailureTracker`'s 5 strikes, and were labeled `agent/attempted`. That label
is excluded from `gmail_query` (`config.toml`), so those threads dropped out
of the poll permanently — never classified, never retried. Production logs did
not survive the container teardown, so the affected-thread count cannot be
reconstructed from logs; **the label itself is the only durable trace**, and
sweeping it is the only recovery path.

## What's in the backlog besides #64 casualties

Not everything under `label:agent/attempted` was dropped by this bug:

- **Proxy-flap give-ups (~2026-07-09 era):** threads abandoned during Gmail
  API proxy outages, unrelated to #64.
- **Newsletter sink faults (a closed window: images built after PR #65 and
  before Wave 2):** PR #65 briefly let a persistent newsletter
  assessment-sink fault strike out to `agent/attempted`. Wave 2 removed that
  (decision D5's sink corollary): a sink fault is shared-cause now — never
  counted, retried forever, never abandoned. So this population exists only if
  an image from inside that window ever ran, and the image deployed as of
  2026-07-30 (built 2026-07-28) predates #65 — meaning the label today still
  means only "classification give-ups", and no future deploy can add sink
  faults to it. **The sweep is still cleanest before the daemon runs again:**
  once it does, fresh classification give-ups accrue under the same label and
  mix into the population you are sorting.

Also note: `FailureTracker` is in-memory. Every daemon restart wiped the
failure counts and the original error logs, so there is no per-thread record
of *why* a given thread was given up on — only the label.

## Recovery steps

1. **Enumerate, while the daemon is stopped and before the next deploy.**
   Search Gmail for `label:agent/attempted`. Expect two populations: #64
   person-thread casualties (roughly, threads from real people since the
   Ollama cutover) and older proxy-flap give-ups.

2. **Re-queue by removing the label.** The mechanism that has worked before:
   remove the `agent/attempted` label — via the Gmail proxy's modify endpoint
   (`modify_message` with `removeLabelIds`), or simply in the Gmail UI (it is
   an ordinary label). One more condition: `gmail_query` is
   `in:inbox -label:agent/processed -label:agent/attempted`, so the thread
   must also still be **in the inbox**. The give-up path never archives, but
   any thread you archived by hand in the meantime needs moving back to the
   inbox as well or it will not re-enter the poll. Removing the label from
   *everything* is safe if in doubt: a thread that still fails for its own
   reasons takes its strikes again and returns to `agent/attempted` — now only
   when the daemon's cycle-level correlation blames the thread rather than a
   shared cause (decision D5), so a re-queued thread failing alongside an
   outage is simply retried instead.

3. **Deploy the fix, then restart the daemon.** `config.toml` is baked into
   the image at build time — the #64 fix is not live until the image is
   rebuilt and the daemon restarted. (The daemon is currently stopped on
   purpose; re-queued threads are inert until it runs again.) Re-queueing
   before the fixed image is up would just re-drop the hard threads.

4. **Watch the drain.** Re-queued threads flow through at
   `max_emails_per_cycle` (default 10) per poll cycle. Watch for
   `LLMContentError` in the logs — with the fix deployed there should be none
   from the local tier; any new ones now carry the response's real
   `finish_reason` and which reasoning field was populated.

## Expectations

- **Labels may differ on reprocess.** Temperature-0 decoding is not stable
  across server states (issue #64, finding 6): a borderline thread can
  legitimately come back `FYI` one cycle and `LOW_PRIORITY` another. A
  re-queued thread getting a different label than it would have in June is
  expected, not a regression.
- **Old threads get labeled late.** `agent/needs-response` on a month-old
  thread from `someone@example.com` may no longer be actionable; the sweep
  restores classification, not timeliness.
