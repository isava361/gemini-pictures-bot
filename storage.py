import time
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

VALID_RATIOS = {"auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
VALID_RESOLUTIONS = {"1K", "2K", "4K"}
VALID_OUTPUT_MODES = {"photo", "file"}

# модели
MODEL_FLASH_ID = "gemini-2.5-flash-image"       # Nano Banana
MODEL_PRO_ID = "gemini-3-pro-image-preview"     # Nano Banana Pro
VALID_MODELS = {MODEL_FLASH_ID, MODEL_PRO_ID}


def model_display_name(model_id: str) -> str:
    if model_id == MODEL_FLASH_ID:
        return "Nano Banana"
    if model_id == MODEL_PRO_ID:
        return "Nano Banana Pro"
    return model_id


@dataclass
class GlobalSettings:
    ratio: str = "1:1"
    resolution: str = "1K"
    model_id: str = MODEL_PRO_ID
    output_mode: str = "photo"


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()


    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            # PRAGMA нужно выставлять на каждом новом соединении
            cur = await self._conn.execute("PRAGMA journal_mode=WAL;")
            await cur.close()
            cur = await self._conn.execute("PRAGMA synchronous=NORMAL;")
            await cur.close()
        return self._conn

    @asynccontextmanager
    async def _db(self):
        # Один SQLite connection + сериализация запросов для безопасности
        async with self._lock:
            db = await self._ensure_conn()
            try:
                yield db
            except Exception:
                # Если где-то упали между write и commit — откатываем транзакцию,
                # иначе можно получить "cannot start a transaction within a transaction"
                try:
                    if getattr(db, "in_transaction", False):
                        await db.rollback()
                finally:
                    raise

    async def aclose(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    async def _has_column(self, db: aiosqlite.Connection, table: str, column: str) -> bool:
        cur = await db.execute(f"PRAGMA table_info({table})")
        rows = await cur.fetchall()
        cols = {r[1] for r in rows}  # r[1] = name
        await cur.close()
        return column in cols

    async def init(self) -> None:
        async with self._db() as db:

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS allowed_users (
                    user_id INTEGER PRIMARY KEY
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS global_settings (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    ratio TEXT NOT NULL,
                    resolution TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    ratio TEXT,
                    resolution TEXT
                )
                """
            )

            # ensure global row exists (старый формат)
            await db.execute(
                """
                INSERT OR IGNORE INTO global_settings (id, ratio, resolution)
                VALUES (1, '1:1', '1K')
                """
            )

            # ---- MIGRATIONS ----
            # 1) model_id в global_settings
            if not await self._has_column(db, "global_settings", "model_id"):
                await db.execute("ALTER TABLE global_settings ADD COLUMN model_id TEXT")
                await db.execute(
                    "UPDATE global_settings SET model_id=? WHERE (model_id IS NULL OR model_id='')",
                    (MODEL_PRO_ID,),
                )

            # 2) model_id в user_settings
            if not await self._has_column(db, "user_settings", "model_id"):
                await db.execute("ALTER TABLE user_settings ADD COLUMN model_id TEXT")
            # 2.1) output_mode в global_settings
            if not await self._has_column(db, "global_settings", "output_mode"):
                await db.execute("ALTER TABLE global_settings ADD COLUMN output_mode TEXT")
                await db.execute(
                    "UPDATE global_settings SET output_mode=? WHERE (output_mode IS NULL OR output_mode='')",
                    ("photo",),
                )
            # 2.2) output_mode в user_settings
            if not await self._has_column(db, "user_settings", "output_mode"):
                await db.execute("ALTER TABLE user_settings ADD COLUMN output_mode TEXT")

            # 3) старая таблица usage_daily (оставляем для совместимости)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_daily(
                    user_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    pro_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, day)
                )
                """
            )

            # 4) НОВАЯ таблица: usage_daily_models (лимиты по модели)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_daily_models(
                    user_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    cnt INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, day, model_id)
                )
                """
            )

            # 5) миграция: переносим pro_count -> usage_daily_models для MODEL_PRO_ID
            await db.execute(
                """
                INSERT OR IGNORE INTO usage_daily_models(user_id, day, model_id, cnt)
                SELECT user_id, day, ?, pro_count
                FROM usage_daily
                WHERE pro_count > 0
                """,
                (MODEL_PRO_ID,),
            )

            # 6) гарантируем, что global_settings.model_id не NULL
            await db.execute(
                "UPDATE global_settings SET model_id=? WHERE id=1 AND (model_id IS NULL OR model_id='')",
                (MODEL_PRO_ID,),
            )
            await db.execute(
                "UPDATE global_settings SET output_mode=? WHERE id=1 AND (output_mode IS NULL OR output_mode='')",
                ("photo",),
            )

            # ---- Token usage + profiles (для /usage) ----
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id    INTEGER PRIMARY KEY,
                    username   TEXT,
                    first_name TEXT,
                    last_name  TEXT,
                    updated_at INTEGER
                )
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            INTEGER NOT NULL,
                    day_key       TEXT NOT NULL,
                    month_key     TEXT NOT NULL,
                    user_id       INTEGER NOT NULL,
                    model_id      TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    text_output_tokens INTEGER NOT NULL DEFAULT 0,
                    thinking_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens  INTEGER NOT NULL DEFAULT 0
                )
                """
            )


            # 6) token_usage: add columns for text/thinking output (if missing)
            if not await self._has_column(db, "token_usage", "text_output_tokens"):
                await db.execute("ALTER TABLE token_usage ADD COLUMN text_output_tokens INTEGER NOT NULL DEFAULT 0")
            if not await self._has_column(db, "token_usage", "thinking_tokens"):
                await db.execute("ALTER TABLE token_usage ADD COLUMN thinking_tokens INTEGER NOT NULL DEFAULT 0")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_day ON token_usage(day_key)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_month ON token_usage(month_key)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_user ON token_usage(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_model ON token_usage(model_id)")

            await db.commit()

    async def seed_admins_as_allowed(self, admin_ids: Iterable[int]) -> None:
        async with self._db() as db:
            for uid in admin_ids:
                await db.execute("INSERT OR IGNORE INTO allowed_users(user_id) VALUES (?)", (int(uid),))
            await db.commit()

    async def is_allowed(self, user_id: int) -> bool:
        async with self._db() as db:
            cur = await db.execute("SELECT 1 FROM allowed_users WHERE user_id=?", (int(user_id),))
            row = await cur.fetchone()
            await cur.close()
            return row is not None

    async def allow(self, user_id: int) -> None:
        async with self._db() as db:
            await db.execute("INSERT OR IGNORE INTO allowed_users(user_id) VALUES (?)", (int(user_id),))
            await db.commit()

    async def deny(self, user_id: int) -> None:
        async with self._db() as db:
            await db.execute("DELETE FROM allowed_users WHERE user_id=?", (int(user_id),))
            await db.execute("DELETE FROM user_settings WHERE user_id=?", (int(user_id),))
            await db.commit()

    async def list_allowed(self) -> List[int]:
        async with self._db() as db:
            cur = await db.execute("SELECT user_id FROM allowed_users ORDER BY user_id")
            rows = await cur.fetchall()
            await cur.close()
            return [int(r[0]) for r in rows]

    async def get_global_settings(self) -> GlobalSettings:
        async with self._db() as db:
            cur = await db.execute("SELECT ratio, resolution, model_id, output_mode FROM global_settings WHERE id=1")
            row = await cur.fetchone()
            await cur.close()
            return GlobalSettings(ratio=row[0], resolution=row[1], model_id=row[2], output_mode=row[3])

    async def set_global_ratio(self, ratio: str) -> None:
        if ratio not in VALID_RATIOS:
            raise ValueError("Invalid ratio")
        async with self._db() as db:
            await db.execute("UPDATE global_settings SET ratio=? WHERE id=1", (ratio,))
            await db.commit()

    async def set_global_resolution(self, resolution: str) -> None:
        resolution = (resolution or "").upper().strip()
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError("Invalid resolution")
        async with self._db() as db:
            await db.execute("UPDATE global_settings SET resolution=? WHERE id=1", (resolution,))
            await db.commit()

    async def set_global_model(self, model_id: str) -> None:
        if model_id not in VALID_MODELS:
            raise ValueError("Invalid model")
        async with self._db() as db:
            await db.execute("UPDATE global_settings SET model_id=? WHERE id=1", (model_id,))
            await db.commit()

    async def set_global_output_mode(self, output_mode: str) -> None:
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError("Invalid output mode")
        async with self._db() as db:
            await db.execute("UPDATE global_settings SET output_mode=? WHERE id=1", (output_mode,))
            await db.commit()

    async def get_effective_settings(self, user_id: int) -> GlobalSettings:
        g = await self.get_global_settings()
        async with self._db() as db:
            cur = await db.execute(
                "SELECT ratio, resolution, model_id, output_mode FROM user_settings WHERE user_id=?",
                (int(user_id),),
            )
            row = await cur.fetchone()
            await cur.close()
            if not row:
                return g
            ratio = row[0] if row[0] else g.ratio
            resolution = row[1] if row[1] else g.resolution
            model_id = row[2] if (len(row) > 2 and row[2]) else g.model_id
            output_mode = row[3] if (len(row) > 3 and row[3]) else g.output_mode
            return GlobalSettings(ratio=ratio, resolution=resolution, model_id=model_id, output_mode=output_mode)

    async def set_user_ratio(self, user_id: int, ratio: str) -> None:
        if ratio not in VALID_RATIOS:
            raise ValueError("Invalid ratio")
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO user_settings(user_id, ratio, resolution, model_id, output_mode)
                VALUES (?, ?, NULL, NULL, NULL)
                ON CONFLICT(user_id) DO UPDATE SET ratio=excluded.ratio
                """,
                (int(user_id), ratio),
            )
            await db.commit()

    async def set_user_resolution(self, user_id: int, resolution: str) -> None:
        resolution = (resolution or "").upper().strip()
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError("Invalid resolution")
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO user_settings(user_id, ratio, resolution, model_id, output_mode)
                VALUES (?, NULL, ?, NULL, NULL)
                ON CONFLICT(user_id) DO UPDATE SET resolution=excluded.resolution
                """,
                (int(user_id), resolution),
            )
            await db.commit()

    async def set_user_model(self, user_id: int, model_id: str) -> None:
        if model_id not in VALID_MODELS:
            raise ValueError("Invalid model")
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO user_settings(user_id, ratio, resolution, model_id, output_mode)
                VALUES (?, NULL, NULL, ?, NULL)
                ON CONFLICT(user_id) DO UPDATE SET model_id=excluded.model_id
                """,
                (int(user_id), model_id),
            )
            await db.commit()

    async def set_user_output_mode(self, user_id: int, output_mode: str) -> None:
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError("Invalid output mode")
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO user_settings(user_id, ratio, resolution, model_id, output_mode)
                VALUES (?, NULL, NULL, NULL, ?)
                ON CONFLICT(user_id) DO UPDATE SET output_mode=excluded.output_mode
                """,
                (int(user_id), output_mode),
            )
            await db.commit()

    # ---- daily usage: per-model ----
    async def get_model_count(self, user_id: int, day: str, model_id: str) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "SELECT cnt FROM usage_daily_models WHERE user_id=? AND day=? AND model_id=?",
                (int(user_id), day, model_id),
            )
            row = await cur.fetchone()
            await cur.close()
            return int(row[0]) if row else 0

    async def inc_model_count(self, user_id: int, day: str, model_id: str, delta: int = 1) -> None:
        if delta <= 0:
            return
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO usage_daily_models(user_id, day, model_id, cnt)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(user_id, day, model_id) DO UPDATE SET cnt = cnt + excluded.cnt
                """,
                (int(user_id), day, model_id, int(delta)),
            )
            await db.commit()

    # ---- compatibility wrappers (старый API "Pro") ----
    async def get_pro_count(self, user_id: int, day: str) -> int:
        return await self.get_model_count(user_id, day, MODEL_PRO_ID)

    async def inc_pro_count(self, user_id: int, day: str, delta: int = 1) -> None:
        await self.inc_model_count(user_id, day, MODEL_PRO_ID, delta)

    # -----------------------------
    # User profile (username/name)
    # -----------------------------
    async def upsert_user_profile(self, user_id: int, username: str, first_name: str, last_name: str) -> None:
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO user_profile(user_id, username, first_name, last_name, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    last_name=excluded.last_name,
                    updated_at=excluded.updated_at
                """,
                (int(user_id), username or "", first_name or "", last_name or "", int(time.time())),
            )
            await db.commit()

    # -----------------------------
    # Token usage logging/reporting
    # -----------------------------
    async def log_token_usage(
        self,
        user_id: int,
        day_key: str,
        month_key: str,
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        text_output_tokens: int = 0,
        thinking_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO token_usage(ts, day_key, month_key, user_id, model_id, prompt_tokens, output_tokens, text_output_tokens, thinking_tokens, total_tokens)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    str(day_key),
                    str(month_key),
                    int(user_id),
                    model_id or "",
                    int(prompt_tokens or 0),
                    int(output_tokens or 0),
                    int(text_output_tokens or 0),
                    int(thinking_tokens or 0),
                    int(total_tokens or 0),
                ),
            )
            await db.commit()

    async def get_usage_by_day(self, day_key: str) -> List[Dict[str, Any]]:
        async with self._db() as db:
            cur = await db.execute(
                """
                SELECT
                    tu.user_id,
                    up.username,
                    up.first_name,
                    up.last_name,
                    SUM(tu.prompt_tokens) AS prompt_tokens,
                    SUM(tu.output_tokens) AS output_tokens,
                    SUM(tu.text_output_tokens) AS text_output_tokens,
                    SUM(tu.thinking_tokens) AS thinking_tokens,
                    SUM(tu.total_tokens)  AS total_tokens
                FROM token_usage tu
                LEFT JOIN user_profile up ON up.user_id = tu.user_id
                WHERE tu.day_key = ?
                GROUP BY tu.user_id
                ORDER BY total_tokens DESC
                """,
                (str(day_key),),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "user_id": int(r[0]),
                    "username": r[1] or "",
                    "first_name": r[2] or "",
                    "last_name": r[3] or "",
                    "prompt_tokens": int(r[4] or 0),
                    "output_tokens": int(r[5] or 0),
                    "text_output_tokens": int(r[6] or 0),
                    "thinking_tokens": int(r[7] or 0),
                    "total_tokens": int(r[8] or 0),
                }
            )
        return out

    async def get_usage_by_month(self, month_key: str) -> List[Dict[str, Any]]:
        async with self._db() as db:
            cur = await db.execute(
                """
                SELECT
                    tu.user_id,
                    up.username,
                    up.first_name,
                    up.last_name,
                    SUM(tu.prompt_tokens) AS prompt_tokens,
                    SUM(tu.output_tokens) AS output_tokens,
                    SUM(tu.text_output_tokens) AS text_output_tokens,
                    SUM(tu.thinking_tokens) AS thinking_tokens,
                    SUM(tu.total_tokens)  AS total_tokens
                FROM token_usage tu
                LEFT JOIN user_profile up ON up.user_id = tu.user_id
                WHERE tu.month_key = ?
                GROUP BY tu.user_id
                ORDER BY total_tokens DESC
                """,
                (str(month_key),),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "user_id": int(r[0]),
                    "username": r[1] or "",
                    "first_name": r[2] or "",
                    "last_name": r[3] or "",
                    "prompt_tokens": int(r[4] or 0),
                    "output_tokens": int(r[5] or 0),
                    "text_output_tokens": int(r[6] or 0),
                    "thinking_tokens": int(r[7] or 0),
                    "total_tokens": int(r[8] or 0),
                }
            )
        return out


    async def get_usage_by_day_models(self, day_key: str) -> List[Dict[str, Any]]:
        """Агрегация токенов по (user_id, model_id) за день."""
        async with self._db() as db:
            cur = await db.execute(
                """
                SELECT
                    tu.user_id,
                    up.username,
                    up.first_name,
                    up.last_name,
                    COALESCE(NULLIF(tu.model_id, ''), 'unknown') AS model_id,
                    SUM(tu.prompt_tokens) AS prompt_tokens,
                    SUM(tu.output_tokens) AS output_tokens,
                    SUM(tu.text_output_tokens) AS text_output_tokens,
                    SUM(tu.thinking_tokens) AS thinking_tokens,
                    SUM(tu.total_tokens)  AS total_tokens
                FROM token_usage tu
                LEFT JOIN user_profile up ON up.user_id = tu.user_id
                WHERE tu.day_key = ?
                GROUP BY tu.user_id, COALESCE(NULLIF(tu.model_id, ''), 'unknown')
                ORDER BY tu.user_id ASC, total_tokens DESC
                """,
                (str(day_key),),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "user_id": int(r[0]),
                    "username": r[1] or "",
                    "first_name": r[2] or "",
                    "last_name": r[3] or "",
                    "model_id": r[4] or "",
                    "prompt_tokens": int(r[5] or 0),
                    "output_tokens": int(r[6] or 0),
                    "text_output_tokens": int(r[7] or 0),
                    "thinking_tokens": int(r[8] or 0),
                    "total_tokens": int(r[9] or 0),
                }
            )
        return out

    async def get_usage_by_month_models(self, month_key: str) -> List[Dict[str, Any]]:
        """Агрегация токенов по (user_id, model_id) за месяц."""
        async with self._db() as db:
            cur = await db.execute(
                """
                SELECT
                    tu.user_id,
                    up.username,
                    up.first_name,
                    up.last_name,
                    COALESCE(NULLIF(tu.model_id, ''), 'unknown') AS model_id,
                    SUM(tu.prompt_tokens) AS prompt_tokens,
                    SUM(tu.output_tokens) AS output_tokens,
                    SUM(tu.text_output_tokens) AS text_output_tokens,
                    SUM(tu.thinking_tokens) AS thinking_tokens,
                    SUM(tu.total_tokens)  AS total_tokens
                FROM token_usage tu
                LEFT JOIN user_profile up ON up.user_id = tu.user_id
                WHERE tu.month_key = ?
                GROUP BY tu.user_id, COALESCE(NULLIF(tu.model_id, ''), 'unknown')
                ORDER BY tu.user_id ASC, total_tokens DESC
                """,
                (str(month_key),),
            )
            rows = await cur.fetchall()
            await cur.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "user_id": int(r[0]),
                    "username": r[1] or "",
                    "first_name": r[2] or "",
                    "last_name": r[3] or "",
                    "model_id": r[4] or "",
                    "prompt_tokens": int(r[5] or 0),
                    "output_tokens": int(r[6] or 0),
                    "text_output_tokens": int(r[7] or 0),
                    "thinking_tokens": int(r[8] or 0),
                    "total_tokens": int(r[9] or 0),
                }
            )
        return out
