# TODO

Remaining improvements identified during code review.

## Dead / Unused Code

- [ ] Remove `_render_usage_rows` function (`bot.py`) — defined but never called, superseded by `_render_usage_rows_models`
- [ ] Remove `usage_daily` table creation and migration in `storage.py` — legacy table, no longer used by any code path
- [ ] Remove `get_pro_count` / `inc_pro_count` compatibility wrappers in `storage.py` — never called
- [ ] Remove `get_usage_by_day` and `get_usage_by_month` in `storage.py` — only the `*_models` variants are used
- [ ] Fix double `output_mode_label` application — `bot.py` calls `output_mode_label(s.output_mode)` when creating `Request`, then calls it again on `req.output_mode` in the "done" status line. Store raw value in `Request`, apply label only at display time

## Code Structure

- [ ] Split `bot.py` (1800+ lines) into modules: `handlers.py` (command/message handlers), `queue.py` (BotState, UserWork, Request, queue logic), `settings_ui.py` (inline keyboard builders, settings callbacks), `usage_report.py` (render functions, formatting)
- [ ] Refactor `handle_settings_callback` (270+ lines of if/elif) into a dispatch table mapping `(action, dest)` to handler functions
- [ ] Rename `GlobalSettings` dataclass in `storage.py` to `Settings` or `EffectiveSettings` — it is used for both global defaults and merged user settings, the current name is misleading

## Performance

- [ ] Deduplicate `get_usage_by_day_models` / `get_usage_by_month_models` in `storage.py` — nearly identical queries, extract a shared private method parameterized by key column
- [ ] Consider a read/write lock for `Storage` — current `asyncio.Lock` serializes all operations including reads; SQLite WAL mode supports concurrent readers
- [ ] `cleanup_state_job` rebuilds entire dicts via comprehensions every 10 minutes — switch to in-place eviction for lower GC pressure at scale

## Robustness

- [ ] Graceful shutdown of in-progress requests — `on_shutdown` closes services and DB but does not cancel or await active `user_worker` tasks; mid-flight requests may error
- [ ] Retry Telegram send failures — if `send_photo`/`send_document` fails transiently after a successful Gemini API call, the generated images are lost
- [ ] Validate `ADMIN_IDS` at startup — log a warning if the env var is empty or contains non-integer values
- [ ] Sanitize table name in `_has_column` (`storage.py`) — uses f-string interpolation for the table name in `PRAGMA table_info({table})`; currently safe (hardcoded callers only) but should use a whitelist or regex check as defense-in-depth

## Testing

- [ ] Add unit tests — no test suite exists; priority targets: `infer_aspect_ratio_from_images`, `cost_usd`, `sanitize_resolution`, `Storage` CRUD operations, settings merge logic
