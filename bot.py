# bot.py
from __future__ import annotations

import asyncio
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Deque, Dict, List, Optional, Tuple, Set
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from telegram import (
    Update,
    InputFile,
    InputMediaPhoto,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest, Forbidden
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from storage import (
    Storage,
    VALID_RATIOS,
    VALID_RESOLUTIONS,
    VALID_MODELS,
    VALID_OUTPUT_MODES,
    MODEL_FLASH_ID,
    MODEL_PRO_ID,
    model_display_name,
)
from gemini_images import GeminiImageService


ALBUM_FLUSH_DELAY_SEC = 1.5
CACHE_TTL_SEC = 10 * 60  # 10 минут
CLEANUP_INTERVAL_SEC = 10 * 60  # уборка кешей раз в 10 минут

# ЛИМИТЫ (для НЕ-админов)
FLASH_DAILY_LIMIT_NONADMIN = 40   # Gemini 2.5 (MODEL_FLASH_ID / Nano Banana)
PRO_DAILY_LIMIT_NONADMIN = 20     # Gemini 3   (MODEL_PRO_ID / Nano Banana Pro)

MODEL_DAILY_LIMITS_NONADMIN = {
    MODEL_FLASH_ID: FLASH_DAILY_LIMIT_NONADMIN,
    MODEL_PRO_ID: PRO_DAILY_LIMIT_NONADMIN,
}

# Admin-only resolutions
ADMIN_ONLY_RESOLUTIONS = {"2K", "4K"}
NONADMIN_RESOLUTION_FALLBACK = "1K"

# -----------------------------
# Pricing (USD per 1M tokens)
#
# We log + bill separately:
# - prompt_tokens (input)
# - text_output_tokens (final text output, if any)
# - thinking_tokens (model internal reasoning tokens)
# - output_tokens (we treat as image output tokens for this bot)
#
# Gemini 3:
#   input:   $2 / 1M
#   text:    $12 / 1M
#   thinking:$12 / 1M
#   image:   $120 / 1M
# Gemini 2.5:
#   input:   $0.30 / 1M
#   image:   $30 / 1M
#   text/thinking rates not provided -> treated as $0 (tokens still logged).
# -----------------------------
MODEL_PRICING_USD_PER_M_TOKENS = {
    # Gemini 3 (Nano Banana Pro)
    MODEL_PRO_ID: {"input": 2.0, "text_out": 12.0, "thinking_out": 12.0, "image_out": 120.0},
    # Gemini 2.5 (Nano Banana)
    MODEL_FLASH_ID: {"input": 0.30, "text_out": 0.0, "thinking_out": 0.0, "image_out": 30.0},
}

def cost_usd(
    model_id: str,
    prompt_tokens: int,
    image_output_tokens: int,
    text_output_tokens: int,
    thinking_tokens: int,
) -> float:
    p = MODEL_PRICING_USD_PER_M_TOKENS.get(model_id)
    if not p:
        return 0.0
    return (
        (float(prompt_tokens or 0) / 1_000_000.0) * float(p.get('input', 0.0))
        + (float(image_output_tokens or 0) / 1_000_000.0) * float(p.get('image_out', 0.0))
        + (float(text_output_tokens or 0) / 1_000_000.0) * float(p.get('text_out', 0.0))
        + (float(thinking_tokens or 0) / 1_000_000.0) * float(p.get('thinking_out', 0.0))
    )

def _fmt_tokens(n: int) -> str:
    n = int(n or 0)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)

def _short_model(model_id: str) -> str:
    if model_id == MODEL_FLASH_ID:
        return "2.5"
    if model_id == MODEL_PRO_ID:
        return "3"
    return model_id or "unknown"

def output_mode_label(output_mode: str) -> str:
    return "file" if output_mode == "file" else "photo"


def sanitize_resolution(res: str, is_admin_flag: bool) -> str:
    v = (res or "").strip().upper()
    if v not in VALID_RESOLUTIONS:
        v = NONADMIN_RESOLUTION_FALLBACK
    if (not is_admin_flag) and (v in ADMIN_ONLY_RESOLUTIONS):
        return NONADMIN_RESOLUTION_FALLBACK
    return v


def daily_limit_for_model(model_id: str) -> Optional[int]:
    return MODEL_DAILY_LIMITS_NONADMIN.get(model_id)


def moscow_day_key() -> str:
    return datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()


def moscow_month_key() -> str:
    return datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m")


# -----------------------------
# Reply helper (fix update.message=None cases)
# -----------------------------
async def reply(
    update: Update,
    context: Optional[ContextTypes.DEFAULT_TYPE],
    text: str,
    **kwargs,
):
    """
    Надёжный ответ: update.message может быть None (например, edited message).
    """
    m = update.effective_message
    if m:
        return await m.reply_text(text, **kwargs)

    chat = update.effective_chat
    if chat and context:
        return await context.bot.send_message(chat_id=chat.id, text=text, **kwargs)

    return None


# -----------------------------
# Utils
# -----------------------------
def try_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def parse_ids_csv(s: str) -> List[int]:
    out: List[int] = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        v = try_int(p)
        if v is None:
            continue
        out.append(v)
    return out


def is_admin(user_id: int, admin_ids: List[int]) -> bool:
    return user_id in admin_ids


async def touch_user_profile(db: Storage, user) -> None:
    if not user:
        return
    try:
        await db.upsert_user_profile(
            user_id=user.id,
            username=getattr(user, "username", "") or "",
            first_name=getattr(user, "first_name", "") or "",
            last_name=getattr(user, "last_name", "") or "",
        )
    except Exception:
        pass


async def ensure_allowed(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    db: Storage,
    admin_ids: List[int],
) -> bool:
    user = update.effective_user
    if not user:
        return False

    uid = user.id
    if is_admin(uid, admin_ids):
        return True

    ok = await db.is_allowed(uid)
    if not ok:
        await reply(
            update,
            context,
            (
                f"Нет доступа.\nВаш Telegram user_id: {uid}\n"
                f"Попросите администратора добавить вас через /allow {uid}"
            ),
        )
    return ok


def pick_best_photo_file_id(message) -> Optional[str]:
    if not message:
        return None
    if message.photo:
        return message.photo[-1].file_id
    if message.document and (message.document.mime_type or "").startswith("image/"):
        return message.document.file_id
    return None


async def download_pil_images(bot, file_ids: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for fid in file_ids:
        tg_file = await bot.get_file(fid)
        data = await tg_file.download_as_bytearray()
        try:
            img = Image.open(BytesIO(data))
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise RuntimeError(f"Не удалось прочитать изображение: {e}")
        images.append(img)
    return images


def make_req_id() -> str:
    # короткий id для callback_data (лимит 64 байта)
    return f"r{int(time.time() * 1000) % 1_000_000_000}{random.randint(100,999)}"


def cancel_markup(req_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⛔️ Отменить", callback_data=f"cancel:{req_id}")]]
    )


async def safe_edit_text(
    bot,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup=None,
) -> None:
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
        )
    except (BadRequest, Forbidden):
        pass
    except Exception:
        pass


async def safe_answer_callback(query, text: str) -> None:
    try:
        await query.answer(text, show_alert=False)
    except Exception:
        pass


# -----------------------------
# Settings UI (inline keyboards)
# -----------------------------
def _chunks(lst, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def settings_main_markup(owner_uid: int, is_admin_flag: bool) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🤖 Model", callback_data=f"st|{owner_uid}|nav|u_model")],
        [
            InlineKeyboardButton("📐 Ratio", callback_data=f"st|{owner_uid}|nav|u_ratio"),
            InlineKeyboardButton("🖼 Resolution", callback_data=f"st|{owner_uid}|nav|u_res"),
        ],
        [InlineKeyboardButton("📤 Output", callback_data=f"st|{owner_uid}|nav|u_output")],
    ]
    if is_admin_flag:
        rows.append([InlineKeyboardButton("🌍 Global model", callback_data=f"st|{owner_uid}|nav|g_model")])
        rows.append(
            [
                InlineKeyboardButton("🌍 Global ratio", callback_data=f"st|{owner_uid}|nav|g_ratio"),
                InlineKeyboardButton("🌍 Global res", callback_data=f"st|{owner_uid}|nav|g_res"),
            ]
        )
        rows.append([InlineKeyboardButton("🌍 Global output", callback_data=f"st|{owner_uid}|nav|g_output")])
        rows.append(
            [
                InlineKeyboardButton("📊 Usage (day)", callback_data=f"st|{owner_uid}|nav|usage_day"),
                InlineKeyboardButton("📅 Usage (month)", callback_data=f"st|{owner_uid}|nav|usage_month"),
            ]
        )
    return InlineKeyboardMarkup(rows)

def admin_usage_markup(owner_uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Usage (day)", callback_data=f"st|{owner_uid}|nav|usage_day"),
                InlineKeyboardButton("📅 Usage (month)", callback_data=f"st|{owner_uid}|nav|usage_month"),
            ],
            [InlineKeyboardButton("◀️ Назад", callback_data=f"st|{owner_uid}|nav|main")],
        ]
    )


def ratio_menu_markup(owner_uid: int, current_ratio: str, scope: str) -> InlineKeyboardMarkup:
    ratios = sorted(list(VALID_RATIOS), key=lambda x: (x != "auto", x))  # auto первым

    buttons: List[InlineKeyboardButton] = []
    for r in ratios:
        label = f"✅ {r}" if r == current_ratio else r
        buttons.append(InlineKeyboardButton(label, callback_data=f"st|{owner_uid}|set|{scope}|ratio|{r}"))

    rows = [list(row) for row in _chunks(buttons, 3)]
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data=f"st|{owner_uid}|nav|main")])
    return InlineKeyboardMarkup(rows)


def res_menu_markup(owner_uid: int, current_res: str, scope: str, is_admin_flag: bool) -> InlineKeyboardMarkup:
    # Для не-админов: скрываем 2K/4K
    current_res = sanitize_resolution(current_res, is_admin_flag)

    res_list = sorted(list(VALID_RESOLUTIONS))
    if not is_admin_flag:
        res_list = [r for r in res_list if r not in ADMIN_ONLY_RESOLUTIONS]

    buttons: List[InlineKeyboardButton] = []
    for r in res_list:
        label = f"✅ {r}" if r == current_res else r
        buttons.append(InlineKeyboardButton(label, callback_data=f"st|{owner_uid}|set|{scope}|res|{r}"))

    rows = [buttons]  # 1 строка
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data=f"st|{owner_uid}|nav|main")])
    return InlineKeyboardMarkup(rows)


def model_menu_markup(owner_uid: int, current_model: str, scope: str) -> InlineKeyboardMarkup:
    choices = [
        (MODEL_FLASH_ID, "Nano Banana"),
        (MODEL_PRO_ID, "Nano Banana Pro"),
    ]
    rows: List[List[InlineKeyboardButton]] = []
    for mid, name in choices:
        label = f"✅ {name}" if mid == current_model else name
        rows.append([InlineKeyboardButton(label, callback_data=f"st|{owner_uid}|set|{scope}|model|{mid}")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data=f"st|{owner_uid}|nav|main")])
    return InlineKeyboardMarkup(rows)


def output_menu_markup(owner_uid: int, current_output: str, scope: str) -> InlineKeyboardMarkup:
    choices = [
        ("photo", "🖼 Picture"),
        ("file", "📎 File"),
    ]
    rows: List[List[InlineKeyboardButton]] = []
    for mode, label in choices:
        title = f"✅ {label}" if mode == current_output else label
        rows.append([InlineKeyboardButton(title, callback_data=f"st|{owner_uid}|set|{scope}|output|{mode}")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data=f"st|{owner_uid}|nav|main")])
    return InlineKeyboardMarkup(rows)


async def render_settings_main_text(db: Storage, user_id: int, is_admin_flag: bool) -> str:
    s = await db.get_effective_settings(user_id)
    eff_res = sanitize_resolution(s.resolution, is_admin_flag)

    lines = [
        "Текущие настройки:",
        f"- model: {model_display_name(s.model_id)} ({s.model_id})",
        f"- ratio: {s.ratio}",
        f"- resolution: {eff_res}",
        f"- output: {output_mode_label(s.output_mode)}",
    ]

    if (not is_admin_flag) and (eff_res != (s.resolution or "").upper()):
        lines.append("🔒 2K/4K доступны только администраторам (вам выставлено 1K).")

    if is_admin_flag:
        gs = await db.get_global_settings()
        lines += [
            "",
            "Глобальные настройки (по умолчанию для всех):",
            f"- global model: {model_display_name(gs.model_id)} ({gs.model_id})",
            f"- global ratio: {gs.ratio}",
            f"- global resolution: {gs.resolution}",
            f"- global output: {output_mode_label(gs.output_mode)}",
        ]

    # Лимиты для не-админов: показываем обе модели
    if not is_admin_flag:
        day = moscow_day_key()
        used_flash = await db.get_model_count(user_id, day, MODEL_FLASH_ID)
        used_pro = await db.get_model_count(user_id, day, MODEL_PRO_ID)

        left_flash = max(0, FLASH_DAILY_LIMIT_NONADMIN - used_flash)
        left_pro = max(0, PRO_DAILY_LIMIT_NONADMIN - used_pro)

        lines += [
            "",
            "Лимиты:",
            f"- Nano Banana (Gemini 2.5): {used_flash}/{FLASH_DAILY_LIMIT_NONADMIN} сегодня (осталось {left_flash})",
            f"- Nano Banana Pro (Gemini 3): {used_pro}/{PRO_DAILY_LIMIT_NONADMIN} сегодня (осталось {left_pro})",
            f"• день {day} • Europe/Moscow",
        ]

    return "\n".join(lines)


async def handle_settings_callback(query, context: ContextTypes.DEFAULT_TYPE, data: str) -> None:
    """
    st|<owner_uid>|nav|main
    st|<owner_uid>|nav|u_model
    st|<owner_uid>|nav|u_ratio
    st|<owner_uid>|nav|u_res
    st|<owner_uid>|nav|g_model
    st|<owner_uid>|nav|g_ratio
    st|<owner_uid>|nav|g_res
    st|<owner_uid>|nav|u_output
    st|<owner_uid>|nav|g_output
    st|<owner_uid>|set|u|model|<model_id>
    st|<owner_uid>|set|u|ratio|<ratio>
    st|<owner_uid>|set|u|res|<1K|2K|4K>
    st|<owner_uid>|set|u|output|<photo|file>
    st|<owner_uid>|set|g|model|<model_id>
    st|<owner_uid>|set|g|ratio|<ratio>
    st|<owner_uid>|set|g|res|<1K|2K|4K>
    st|<owner_uid>|set|g|output|<photo|file>
    """
    parts = data.split("|")
    if len(parts) < 4:
        await safe_answer_callback(query, "Некорректные данные.")
        return

    db: Storage = context.application.bot_data["db"]
    admin_ids: List[int] = context.application.bot_data["admin_ids"]

    owner_uid = try_int(parts[1])
    if owner_uid is None:
        await safe_answer_callback(query, "Некорректный user_id.")
        return

    caller_uid = query.from_user.id
    caller_is_admin = is_admin(caller_uid, admin_ids)
    owner_is_admin = is_admin(owner_uid, admin_ids)

    if caller_uid != owner_uid and not caller_is_admin:
        await safe_answer_callback(query, "Это меню не для вас.")
        return

    if not query.message:
        await safe_answer_callback(query, "Нет сообщения для редактирования.")
        return

    chat_id = query.message.chat_id
    message_id = query.message.message_id

    action = parts[2]

    if action == "nav":
        dest = parts[3]

        if dest == "main":
            text = await render_settings_main_text(db, owner_uid, owner_is_admin)
            await safe_edit_text(context.bot, chat_id, message_id, text, reply_markup=settings_main_markup(owner_uid, owner_is_admin))
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "u_model":
            s = await db.get_effective_settings(owner_uid)
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите модель (персонально):",
                reply_markup=model_menu_markup(owner_uid, s.model_id, scope="u"),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "u_ratio":
            s = await db.get_effective_settings(owner_uid)
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите ratio (персонально):",
                reply_markup=ratio_menu_markup(owner_uid, s.ratio, scope="u"),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "u_res":
            s = await db.get_effective_settings(owner_uid)
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите resolution (персонально):",
                reply_markup=res_menu_markup(owner_uid, s.resolution, scope="u", is_admin_flag=owner_is_admin),
            )
            await safe_answer_callback(query, "Ок.")
            return
        if dest == "u_output":
            s = await db.get_effective_settings(owner_uid)
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите формат отправки (персонально):",
                reply_markup=output_menu_markup(owner_uid, s.output_mode, scope="u"),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "g_model":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return
            gs = await db.get_global_settings()
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите GLOBAL модель:",
                reply_markup=model_menu_markup(owner_uid, gs.model_id, scope="g"),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "g_ratio":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return
            gs = await db.get_global_settings()
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите GLOBAL ratio:",
                reply_markup=ratio_menu_markup(owner_uid, gs.ratio, scope="g"),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "g_res":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return
            gs = await db.get_global_settings()
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите GLOBAL resolution:",
                reply_markup=res_menu_markup(owner_uid, gs.resolution, scope="g", is_admin_flag=True),
            )
            await safe_answer_callback(query, "Ок.")
            return
        if dest == "g_output":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return
            gs = await db.get_global_settings()
            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                "Выберите GLOBAL формат отправки:",
                reply_markup=output_menu_markup(owner_uid, gs.output_mode, scope="g"),
            )
            await safe_answer_callback(query, "Ок.")
            return
            
        if dest == "usage_day":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return

            day = moscow_day_key()
            rows = await db.get_usage_by_day_models(day)
            text = _render_usage_rows_models(f"Токены за день {day} (Europe/Moscow):", rows, show_cost=False)

            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                text,
                reply_markup=admin_usage_markup(owner_uid),
            )
            await safe_answer_callback(query, "Ок.")
            return

        if dest == "usage_month":
            if not caller_is_admin:
                await safe_answer_callback(query, "Только для администратора.")
                return

            month = moscow_month_key()
            rows = await db.get_usage_by_month_models(month)
            text = _render_usage_rows_models(f"Траты за месяц {month} (USD, Europe/Moscow):", rows, show_cost=True)

            await safe_edit_text(
                context.bot,
                chat_id,
                message_id,
                text,
                reply_markup=admin_usage_markup(owner_uid),
            )
            await safe_answer_callback(query, "Ок.")
            return


        await safe_answer_callback(query, "Неизвестная навигация.")
        return

    if action == "set":
        if len(parts) < 6:
            await safe_answer_callback(query, "Некорректные данные.")
            return

        scope = parts[3]   # u / g
        field = parts[4]   # model / ratio / res
        value = parts[5]

        if scope == "g" and not caller_is_admin:
            await safe_answer_callback(query, "Только для администратора.")
            return

        if field == "model":
            if value not in VALID_MODELS:
                await safe_answer_callback(query, "Неверная модель.")
                return
            if scope == "u":
                await db.set_user_model(owner_uid, value)
            else:
                await db.set_global_model(value)

        elif field == "ratio":
            if value not in VALID_RATIOS:
                await safe_answer_callback(query, "Неверный ratio.")
                return
            if scope == "u":
                await db.set_user_ratio(owner_uid, value)
            else:
                await db.set_global_ratio(value)

        elif field == "res":
            v = value.strip().upper()
            if v not in VALID_RESOLUTIONS:
                await safe_answer_callback(query, "Неверное разрешение.")
                return

            # ВАЖНО: 2K/4K разрешаем только если владелец (owner_uid) — админ
            if scope == "u" and (not owner_is_admin) and (v in ADMIN_ONLY_RESOLUTIONS):
                await safe_answer_callback(query, "🔒 2K/4K доступны только администраторам.")
                return

            if scope == "u":
                await db.set_user_resolution(owner_uid, v)
            else:
                await db.set_global_resolution(v)

        elif field == "output":
            v = value.strip().lower()
            if v not in VALID_OUTPUT_MODES:
                await safe_answer_callback(query, "Неверный формат.")
                return
            if scope == "u":
                await db.set_user_output_mode(owner_uid, v)
            else:
                await db.set_global_output_mode(v)

        else:
            await safe_answer_callback(query, "Неизвестное поле.")
            return

        text = await render_settings_main_text(db, owner_uid, owner_is_admin)
        await safe_edit_text(context.bot, chat_id, message_id, text, reply_markup=settings_main_markup(owner_uid, owner_is_admin))
        await safe_answer_callback(query, "✅ Сохранено")
        return

    await safe_answer_callback(query, "Неизвестное действие.")


# -----------------------------
# Auto aspect-ratio inference
# -----------------------------
SUPPORTED_RATIOS_FOR_AUTO = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]


def _ratio_value(r: str) -> float:
    a, b = r.split(":")
    return float(a) / float(b)


def infer_aspect_ratio_from_images(images: List[Image.Image]) -> str:
    """
    Авто-выбор ближайшего допустимого aspect ratio по исходной фотке.
    Для альбома берём первую картинку (обычно формат единый).
    """
    if not images:
        return "9:16"

    w, h = images[0].size
    if w <= 0 or h <= 0:
        return "1:1"

    target = w / h

    best = "1:1"
    best_score = float("inf")
    for r in SUPPORTED_RATIOS_FOR_AUTO:
        rv = _ratio_value(r)
        score = abs(math.log(target) - math.log(rv))
        if score < best_score:
            best_score = score
            best = r
    return best


# -----------------------------
# Queue + cancellation
# -----------------------------
@dataclass
class CachedInputs:
    file_ids: List[str]
    ts: float


@dataclass
class PendingAlbum:
    chat_id: int
    user_id: int
    file_ids: List[str]
    message_ids: List[int]
    caption: Optional[str]
    reply_to_message_id: Optional[int]
    ts: float


@dataclass
class Request:
    req_id: str
    chat_id: int
    user_id: int
    prompt: str
    file_ids: Optional[List[str]]
    status_message_id: int
    reply_to_message_id: Optional[int]
    created_at: float

    # фиксируем настройки на момент постановки в очередь
    model_id: str
    ratio: str
    resolution: str
    output_mode: str


class UserWork:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.queue: Deque[Request] = deque()
        self.worker_task: Optional[asyncio.Task] = None
        self.current_req: Optional[Request] = None
        self.current_task: Optional[asyncio.Task] = None


class BotState:
    def __init__(self):
        self.cached: Dict[int, CachedInputs] = {}
        self.pending_albums: Dict[Tuple[int, str], PendingAlbum] = {}
        self.album_jobs: Dict[Tuple[int, str], object] = {}

        self.reply_inputs: Dict[Tuple[int, int], CachedInputs] = {}
        self.media_group_inputs: Dict[Tuple[int, str], CachedInputs] = {}

        self.user_work: Dict[int, UserWork] = {}
        self.req_index: Dict[str, Request] = {}

        self.cancelled: Set[str] = set()


def get_user_work(state: BotState, user_id: int) -> UserWork:
    uw = state.user_work.get(user_id)
    if not uw:
        uw = UserWork()
        state.user_work[user_id] = uw
    return uw


async def enqueue_request(
    app: Application,
    state: BotState,
    db: Storage,
    admin_ids: List[int],
    chat_id: int,
    user_id: int,
    prompt: str,
    file_ids: Optional[List[str]],
    reply_to_message_id: Optional[int],
) -> None:
    s = await db.get_effective_settings(user_id)
    admin_flag = is_admin(user_id, admin_ids)
    eff_res = sanitize_resolution(s.resolution, admin_flag)
    req_id = make_req_id()

    uw = get_user_work(state, user_id)

    # лимит на выбранную модель для не-админов:
    # учитываем уже использованное + то, что уже в очереди/выполняется ЭТОЙ ЖЕ моделью
    async with uw.lock:
        position = len(uw.queue) + (1 if uw.current_req else 0) + 1
        queued_same_model = 0
        if not admin_flag:
            queued_same_model = sum(1 for r in uw.queue if r.model_id == s.model_id)
            if uw.current_req and uw.current_req.model_id == s.model_id:
                queued_same_model += 1

    if not admin_flag:
        limit = daily_limit_for_model(s.model_id)
        if limit is not None:
            used = await db.get_model_count(user_id, moscow_day_key(), s.model_id)
            if used + queued_same_model >= limit:
                await app.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=reply_to_message_id,
                    text=(
                        f"🚫 Лимит {model_display_name(s.model_id)}: {limit} запросов в день.\n"
                        f"Сегодня уже использовано: {used}/{limit}.\n"
                        f"Переключись на другую модель в /settings → Model."
                    ),
                )
                return

    one_line = (
        f"⏳ В очереди: #{position} • {model_display_name(s.model_id)} • "
        f"ratio {s.ratio} • res {eff_res} • output {output_mode_label(s.output_mode)}"
    )
    status_msg = await app.bot.send_message(
        chat_id=chat_id,
        text=one_line,
        reply_to_message_id=reply_to_message_id,
        reply_markup=cancel_markup(req_id),
    )

    req = Request(
        req_id=req_id,
        chat_id=chat_id,
        user_id=user_id,
        prompt=prompt,
        file_ids=file_ids,
        status_message_id=status_msg.message_id,
        reply_to_message_id=reply_to_message_id,
        created_at=time.time(),
        model_id=s.model_id,
        ratio=s.ratio,
        resolution=eff_res,  # уже “клампнутый” res
        output_mode=output_mode_label(s.output_mode),
    )
    state.req_index[req_id] = req

    async with uw.lock:
        uw.queue.append(req)
        if uw.worker_task is None or uw.worker_task.done():
            uw.worker_task = app.create_task(user_worker(app, state, db, admin_ids, user_id))


async def user_worker(
    app: Application,
    state: BotState,
    db: Storage,
    admin_ids: List[int],
    user_id: int,
) -> None:
    uw = get_user_work(state, user_id)

    while True:
        async with uw.lock:
            if uw.current_req is None and not uw.queue:
                uw.worker_task = None
                return

            if uw.current_req is None and uw.queue:
                uw.current_req = uw.queue.popleft()
                req = uw.current_req
            else:
                req = uw.current_req

        if not req:
            await asyncio.sleep(0.05)
            continue

        if req.req_id in state.cancelled:
            await safe_edit_text(app.bot, req.chat_id, req.status_message_id, "⛔️ Отменено.", reply_markup=None)
            state.req_index.pop(req.req_id, None)
            state.cancelled.discard(req.req_id)
            async with uw.lock:
                uw.current_req = None
            continue

        uw.current_task = app.create_task(handle_request(app, state, db, admin_ids, req))
        try:
            await uw.current_task
        except asyncio.CancelledError:
            pass
        finally:
            uw.current_task = None
            async with uw.lock:
                uw.current_req = None


async def handle_request(
    app: Application,
    state: BotState,
    db: Storage,
    admin_ids: List[int],
    req: Request,
) -> None:
    t0 = time.time()

    if req.req_id in state.cancelled:
        await safe_edit_text(app.bot, req.chat_id, req.status_message_id, "⛔️ Отменено.", reply_markup=None)
        state.req_index.pop(req.req_id, None)
        state.cancelled.discard(req.req_id)
        return

    # проверка доступа на всякий случай (если админ удалил user в процессе)
    if not (is_admin(req.user_id, admin_ids) or await db.is_allowed(req.user_id)):
        await safe_edit_text(app.bot, req.chat_id, req.status_message_id, "⛔️ Нет доступа (вас удалили из whitelist).", reply_markup=None)
        state.req_index.pop(req.req_id, None)
        state.cancelled.discard(req.req_id)
        return

    # лимит на момент исполнения (чтобы очередь не обходила лимит)
    if not is_admin(req.user_id, admin_ids):
        day = moscow_day_key()
        limit = daily_limit_for_model(req.model_id)
        if limit is not None:
            used = await db.get_model_count(req.user_id, day, req.model_id)
            if used >= limit:
                await safe_edit_text(
                    app.bot,
                    req.chat_id,
                    req.status_message_id,
                    (
                        f"🚫 Лимит {model_display_name(req.model_id)}: {limit} запросов/день.\n"
                        f"Сегодня уже использовано: {used}/{limit}.\n"
                        f"Переключись на другую модель в /settings → Model."
                    ),
                    reply_markup=None,
                )
                state.req_index.pop(req.req_id, None)
                state.cancelled.discard(req.req_id)
                return

            # резервируем попытку
            await db.inc_model_count(req.user_id, day, req.model_id, 1)

    svcs: Dict[str, GeminiImageService] = app.bot_data["svcs"]
    svc: GeminiImageService = svcs[req.model_id]

    try:
        await safe_edit_text(
            app.bot,
            req.chat_id,
            req.status_message_id,
            f"⏳ [1/3] Подготовка… • {model_display_name(req.model_id)}",
            reply_markup=cancel_markup(req.req_id),
        )

        input_images = None
        if req.file_ids:
            await safe_edit_text(
                app.bot,
                req.chat_id,
                req.status_message_id,
                f"⏳ [1/3] Скачиваю изображения… • {model_display_name(req.model_id)}",
                reply_markup=cancel_markup(req.req_id),
            )
            input_images = await download_pil_images(app.bot, req.file_ids)

        chosen_ratio = req.ratio
        if chosen_ratio == "auto":
            chosen_ratio = infer_aspect_ratio_from_images(input_images) if input_images else "9:16"

        # Defense-in-depth: ещё раз “клампим” res перед вызовом API
        effective_res = sanitize_resolution(req.resolution, is_admin(req.user_id, admin_ids))

        await safe_edit_text(
            app.bot,
            req.chat_id,
            req.status_message_id,
            (
                f"⏳ [2/3] Запрос в Gemini… • {model_display_name(req.model_id)} • "
                f"ratio {chosen_ratio} • res {effective_res}"
            ),
            reply_markup=cancel_markup(req.req_id),
        )
        await app.bot.send_chat_action(chat_id=req.chat_id, action=ChatAction.TYPING)

        # ВАЖНО: используем generate_or_edit_with_usage() из твоего gemini_images.py
        images, usage = await svc.generate_or_edit_with_usage(
            prompt=req.prompt,
            aspect_ratio=chosen_ratio,
            image_size=effective_res,
            input_images=input_images,
        )

        # логируем токены (не валим запрос, если лог не удался)
        try:
            await db.log_token_usage(
                user_id=req.user_id,
                day_key=moscow_day_key(),
                month_key=moscow_month_key(),
                model_id=req.model_id,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                # For image models we store image output tokens in output_tokens (legacy column name).
                output_tokens=int(getattr(usage, "image_output_tokens", getattr(usage, "output_tokens", 0)) or 0),
                text_output_tokens=int(getattr(usage, "text_output_tokens", 0) or 0),
                thinking_tokens=int(getattr(usage, "thinking_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            )
        except Exception:
            pass

        await safe_edit_text(
            app.bot,
            req.chat_id,
            req.status_message_id,
            f"⏳ [3/3] Загружаю результат ({len(images)})… • {model_display_name(req.model_id)}",
            reply_markup=cancel_markup(req.req_id),
        )
        await app.bot.send_chat_action(chat_id=req.chat_id, action=ChatAction.UPLOAD_PHOTO)

        await send_images(app, req.chat_id, images, req.output_mode)

        dt = time.time() - t0
        done_line = (
            f"✅ Готово • {len(images)} шт • {dt:.1f}s • {model_display_name(req.model_id)} • "
            f"ratio {chosen_ratio} • {effective_res} • output {output_mode_label(req.output_mode)}"
        )
        await safe_edit_text(app.bot, req.chat_id, req.status_message_id, done_line, reply_markup=None)

    except asyncio.CancelledError:
        await safe_edit_text(app.bot, req.chat_id, req.status_message_id, "⛔️ Отменено.", reply_markup=None)
        raise
    except Exception as e:
        await safe_edit_text(app.bot, req.chat_id, req.status_message_id, f"❌ Ошибка: {e}", reply_markup=None)
    finally:
        state.req_index.pop(req.req_id, None)
        state.cancelled.discard(req.req_id)


async def send_images(app: Application, chat_id: int, images_png: List[bytes], output_mode: str) -> None:
    """
    Надёжная отправка результатов:
    - пытаемся отправлять пачками через send_media_group (до 10 фото, каждое <= ~9MB),
    - при ошибке Telegram "Can't parse inputmedia: media not found" — fallback на одиночные send_photo/send_document,
    - для >10 изображений делим на чанки по 10.
    """
    if not images_png:
        return

    MAX_PHOTO_BYTES = 9_000_000
    MAX_MEDIA_GROUP = 10

    def chunks(seq: List[bytes], n: int):
        for i in range(0, len(seq), n):
            yield i, seq[i : i + n]

    use_photo = output_mode != "file"

    for base_index, chunk in chunks(images_png, MAX_MEDIA_GROUP):
        # 1) Попытка media_group (только если в чанке >1 и все <= лимита)
        if use_photo and len(chunk) > 1 and all(len(b) <= MAX_PHOTO_BYTES for b in chunk):
            media: List[InputMediaPhoto] = []
            for j, b in enumerate(chunk):
                idx = base_index + j + 1
                bio = BytesIO(b)
                bio.seek(0)
                file = InputFile(bio, filename=f"image_{idx}.png")
                media.append(InputMediaPhoto(media=file))

            try:
                await app.bot.send_media_group(chat_id=chat_id, media=media)
                continue
            except BadRequest as e:
                msg = str(e)
                if ("Can't parse inputmedia" not in msg) and ("media not found" not in msg):
                    raise
                # иначе — fallback на одиночные отправки ниже

        # 2) Fallback: по одному (и для 1 картинки, и для больших, и если media_group не прошёл)
        for j, b in enumerate(chunk):
            idx = base_index + j + 1
            bio = BytesIO(b)
            bio.seek(0)
            file = InputFile(bio, filename=f"image_{idx}.png")

            if use_photo and len(b) <= MAX_PHOTO_BYTES:
                await app.bot.send_photo(chat_id=chat_id, photo=file)
            else:
                await app.bot.send_document(chat_id=chat_id, document=file)


# -----------------------------
# Callback: settings + cancel
# -----------------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data

    if data.startswith("st|"):
        await handle_settings_callback(query, context, data)
        return

    if not data.startswith("cancel:"):
        return

    req_id = data.split(":", 1)[1].strip()
    state: BotState = context.application.bot_data["state"]
    admin_ids: List[int] = context.application.bot_data["admin_ids"]

    req = state.req_index.get(req_id)
    if not req:
        await safe_answer_callback(query, "Уже завершено/не найдено.")
        return

    caller = query.from_user.id
    if caller != req.user_id and not is_admin(caller, admin_ids):
        await safe_answer_callback(query, "Нельзя отменять чужие запросы.")
        return

    state.cancelled.add(req_id)

    uw = get_user_work(state, req.user_id)
    cancel_current = False
    removed_from_queue = False

    async with uw.lock:
        if uw.current_req and uw.current_req.req_id == req_id:
            if uw.current_task and not uw.current_task.done():
                uw.current_task.cancel()
            cancel_current = True
        else:
            newq = deque()
            for r in uw.queue:
                if r.req_id == req_id:
                    removed_from_queue = True
                    continue
                newq.append(r)
            uw.queue = newq

    if cancel_current:
        await safe_answer_callback(query, "Отменяю текущий запрос…")
        return

    if removed_from_queue:
        await safe_edit_text(context.bot, req.chat_id, req.status_message_id, "⛔️ Отменено (снято из очереди).", reply_markup=None)
        state.req_index.pop(req_id, None)
        state.cancelled.discard(req_id)
        await safe_answer_callback(query, "Убрано из очереди.")
        return

    await safe_answer_callback(query, "Ок, отменяю…")


# -----------------------------
# Commands
# -----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: Storage = context.application.bot_data["db"]
    await touch_user_profile(db, update.effective_user)

    uid = update.effective_user.id
    await reply(
        update,
        context,
        "Привет! Я бот для генерации/редактирования изображений.\n\n"
        "Создание: отправь текст.\n"
        "Редактирование: отправь фото/альбом, потом текст (или caption у фото).\n"
        "Можно: ответить (reply) текстом на фото/альбом — я возьму их как input.\n\n"
        "Настройки:\n"
        "• /settings — модель (Nano Banana / Nano Banana Pro), ratio, resolution, output\n\n"
        f"Твой user_id: {uid}\n\n"
        "Команды:\n"
        "/settings — текущие настройки\n"
        "/whoami — показать user_id\n\n"
    )


async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: Storage = context.application.bot_data["db"]
    await touch_user_profile(db, update.effective_user)
    await reply(update, context, f"Ваш Telegram user_id: {update.effective_user.id}")


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: Storage = context.application.bot_data["db"]
    admin_ids: List[int] = context.application.bot_data["admin_ids"]

    await touch_user_profile(db, update.effective_user)

    if not await ensure_allowed(update, context, db, admin_ids):
        return

    uid = update.effective_user.id
    admin_flag = is_admin(uid, admin_ids)

    text = await render_settings_main_text(db, uid, admin_flag)
    await reply(update, context, text, reply_markup=settings_main_markup(uid, admin_flag))


# --- Admin commands ---
async def admin_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not is_admin(update.effective_user.id, context.application.bot_data["admin_ids"]):
        await reply(update, context, "Только для администратора.")
        return False
    return True


async def cmd_allow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update, context):
        return
    db: Storage = context.application.bot_data["db"]

    if not context.args:
        await reply(update, context, "Использование: /allow <telegram_user_id>")
        return
    uid = try_int(context.args[0])
    if uid is None:
        await reply(update, context, "Нужно число. Пример: /allow 123456789")
        return
    await db.allow(uid)
    await reply(update, context, f"Добавил в whitelist: {uid}")


async def cmd_deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update, context):
        return
    db: Storage = context.application.bot_data["db"]

    if not context.args:
        await reply(update, context, "Использование: /deny <telegram_user_id>")
        return
    uid = try_int(context.args[0])
    if uid is None:
        await reply(update, context, "Нужно число. Пример: /deny 123456789")
        return
    await db.deny(uid)
    await reply(update, context, f"Удалил из whitelist: {uid}")


async def cmd_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update, context):
        return
    db: Storage = context.application.bot_data["db"]
    users = await db.list_allowed()
    txt = "Whitelist пуст." if not users else "Whitelist:\n" + "\n".join(map(str, users))
    await reply(update, context, txt)


def _fmt_user_row(r: dict) -> str:
    uid = r.get("user_id")
    uname = (r.get("username") or "").strip()
    name = " ".join([(r.get("first_name") or "").strip(), (r.get("last_name") or "").strip()]).strip()
    who = f"@{uname}" if uname else (name or "—")
    if name and uname:
        who = f"@{uname} ({name})"
    return f"{who} [{uid}]"


def _render_usage_rows(title: str, rows: List[dict]) -> str:
    if not rows:
        return f"{title}\n(данных нет)"

    # чтобы не упереться в лимит 4096 символов
    max_lines = 45
    lines = [title]
    shown = 0

    rest_prompt = rest_out = rest_total = 0
    for i, r in enumerate(rows):
        if shown < max_lines:
            lines.append(
                f"- {_fmt_user_row(r)}: total {r['total_tokens']} "
                f"(prompt {r['prompt_tokens']}, out {r['output_tokens']})"
            )
            shown += 1
        else:
            rest_prompt += int(r.get("prompt_tokens") or 0)
            rest_out += int(r.get("output_tokens") or 0)
            rest_total += int(r.get("total_tokens") or 0)

    if len(rows) > shown:
        lines.append(f"…и ещё {len(rows)-shown} пользователей: total {rest_total} (prompt {rest_prompt}, out {rest_out})")

    return "\n".join(lines)


def _render_usage_rows_models(title: str, rows: List[dict], *, show_cost: bool) -> str:
    """rows: aggregated per (user_id, model_id)."""
    if not rows:
        return f"{title}\n(данных нет)"

    # group by user_id
    users: Dict[int, dict] = {}
    for r in rows:
        uid = int(r.get("user_id") or 0)
        if uid not in users:
            users[uid] = {
                "user_id": uid,
                "username": r.get("username") or "",
                "first_name": r.get("first_name") or "",
                "last_name": r.get("last_name") or "",
                "models": {},
            }
        mid = (r.get("model_id") or "unknown").strip() or "unknown"
        users[uid]["models"][mid] = {
            "prompt_tokens": int(r.get("prompt_tokens") or 0),
            "output_tokens": int(r.get("output_tokens") or 0),
            "text_output_tokens": int(r.get("text_output_tokens") or 0),
            "thinking_tokens": int(r.get("thinking_tokens") or 0),
            "total_tokens": int(r.get("total_tokens") or 0),
        }

    # compute totals + sort
    user_list = []
    for u in users.values():
        tot_prompt = tot_out = tot_text = tot_thinking = tot_total = 0
        tot_cost = 0.0
        for mid, st in u["models"].items():
            tot_prompt += st["prompt_tokens"]
            tot_out += st["output_tokens"]
            tot_text += st.get("text_output_tokens", 0)
            tot_thinking += st.get("thinking_tokens", 0)
            tot_total += st["total_tokens"]
            tot_cost += cost_usd(mid, st["prompt_tokens"], st["output_tokens"], st.get("text_output_tokens", 0), st.get("thinking_tokens", 0))
        u["tot_prompt"] = tot_prompt
        u["tot_out"] = tot_out
        u["tot_text"] = tot_text
        u["tot_thinking"] = tot_thinking
        u["tot_total"] = tot_total
        u["tot_cost"] = tot_cost
        user_list.append(u)

    if show_cost:
        user_list.sort(key=lambda x: x["tot_cost"], reverse=True)
    else:
        user_list.sort(key=lambda x: x["tot_total"], reverse=True)

    # render (avoid 4096 limit)
    max_lines = 35 if show_cost else 45
    lines = [title]

    # Small summary line to make the report easier to scan.
    model_totals: Dict[str, dict] = {}
    for u in user_list:
        for mid, st in u["models"].items():
            mt = model_totals.setdefault(mid, {"prompt": 0, "img": 0, "text": 0, "thinking": 0, "total": 0, "cost": 0.0})
            mt["prompt"] += int(st.get("prompt_tokens") or 0)
            mt["img"] += int(st.get("output_tokens") or 0)
            mt["text"] += int(st.get("text_output_tokens") or 0)
            mt["thinking"] += int(st.get("thinking_tokens") or 0)
            mt["total"] += int(st.get("total_tokens") or 0)
            mt["cost"] += float(cost_usd(mid, int(st.get("prompt_tokens") or 0), int(st.get("output_tokens") or 0), int(st.get("text_output_tokens") or 0), int(st.get("thinking_tokens") or 0)))

    if model_totals:
        mids = list(model_totals.keys())
        mids.sort(key=lambda mid: (mid not in (MODEL_FLASH_ID, MODEL_PRO_ID), _short_model(mid)))
        segs = []
        for mid in mids:
            mt = model_totals[mid]
            if show_cost:
                segs.append(
                    f"{_short_model(mid)} ${mt['cost']:.2f} (in {_fmt_tokens(mt['prompt'])} / img {_fmt_tokens(mt['img'])} / txt {_fmt_tokens(mt['text'])} / th {_fmt_tokens(mt['thinking'])})"
                )
            else:
                segs.append(
                    f"{_short_model(mid)} in {_fmt_tokens(mt['prompt'])} / img {_fmt_tokens(mt['img'])} / txt {_fmt_tokens(mt['text'])} / th {_fmt_tokens(mt['thinking'])}"
                )
        lines.append("По моделям: " + " | ".join(segs))

    shown = 0
    rest = {"prompt": 0, "out": 0, "text": 0, "thinking": 0, "total": 0, "cost": 0.0}
    for u in user_list:
        if shown >= max_lines:
            rest["prompt"] += int(u["tot_prompt"])
            rest["out"] += int(u["tot_out"])
            rest["text"] += int(u.get("tot_text", 0))
            rest["thinking"] += int(u.get("tot_thinking", 0))
            rest["total"] += int(u["tot_total"])
            rest["cost"] += float(u["tot_cost"])
            continue

        who = _fmt_user_row(u)

        parts = []
        # stable model ordering: 2.5 then 3 then others
        model_ids = list(u["models"].keys())
        model_ids.sort(key=lambda mid: (mid not in (MODEL_FLASH_ID, MODEL_PRO_ID), _short_model(mid)))

        for mid in model_ids:
            st = u["models"][mid]
            seg = (f"{_short_model(mid)} in {_fmt_tokens(st['prompt_tokens'])} / img {_fmt_tokens(st['output_tokens'])} / txt {_fmt_tokens(st.get('text_output_tokens', 0))} / th {_fmt_tokens(st.get('thinking_tokens', 0))}")
            if show_cost:
                seg += f" (${cost_usd(mid, st['prompt_tokens'], st['output_tokens'], st.get('text_output_tokens', 0), st.get('thinking_tokens', 0)):.2f})"
            parts.append(seg)

        if show_cost:
            lines.append(
                f"- {who}: ${u['tot_cost']:.2f} • " + " | ".join(parts)
            )
        else:
            lines.append(
                f"- {who}: total {_fmt_tokens(u['tot_total'])} • "
                f"in {_fmt_tokens(u['tot_prompt'])} / img {_fmt_tokens(u['tot_out'])} / txt {_fmt_tokens(u.get('tot_text', 0))} / th {_fmt_tokens(u.get('tot_thinking', 0))} • "
                + " | ".join(parts)
            )
        shown += 1

    if len(user_list) > shown:
        if show_cost:
            lines.append(
                f"…и ещё {len(user_list)-shown} пользователей: ${rest['cost']:.2f} (total {_fmt_tokens(rest['total'])})"
            )
        else:
            lines.append(
                f"…и ещё {len(user_list)-shown} пользователей: total {_fmt_tokens(rest['total'])} • "
                f"in {_fmt_tokens(rest['prompt'])} / img {_fmt_tokens(rest['out'])} / txt {_fmt_tokens(rest['text'])} / th {_fmt_tokens(rest['thinking'])}"
            )

    # footer totals
    grand_prompt = sum(u["tot_prompt"] for u in user_list)
    grand_out = sum(u["tot_out"] for u in user_list)
    grand_text = sum(u.get("tot_text", 0) for u in user_list)
    grand_thinking = sum(u.get("tot_thinking", 0) for u in user_list)
    grand_total = sum(u["tot_total"] for u in user_list)
    grand_cost = sum(float(u["tot_cost"]) for u in user_list)

    if show_cost:
        lines.append("")
        lines.append(f"Итого: ${grand_cost:.2f} • in {_fmt_tokens(grand_prompt)} / img {_fmt_tokens(grand_out)} / txt {_fmt_tokens(grand_text)} / th {_fmt_tokens(grand_thinking)}")
    else:
        lines.append("")
        lines.append(f"Итого: total {_fmt_tokens(grand_total)} • in {_fmt_tokens(grand_prompt)} / img {_fmt_tokens(grand_out)} / txt {_fmt_tokens(grand_text)} / th {_fmt_tokens(grand_thinking)}")

    return "\n".join(lines)


async def cmd_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update, context):
        return
    db: Storage = context.application.bot_data["db"]

    day = moscow_day_key()
    rows = await db.get_usage_by_day_models(day)

    text = _render_usage_rows_models(f"Токены за день {day} (Europe/Moscow):", rows, show_cost=False)
    await reply(update, context, text)


async def cmd_usage_month(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update, context):
        return
    db: Storage = context.application.bot_data["db"]

    month = moscow_month_key()
    rows = await db.get_usage_by_month_models(month)

    text = _render_usage_rows_models(f"Траты за месяц {month} (USD, Europe/Moscow):", rows, show_cost=True)
    await reply(update, context, text)


# -----------------------------
# Handlers: text / images (single + album)
# -----------------------------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: Storage = context.application.bot_data["db"]
    state: BotState = context.application.bot_data["state"]
    admin_ids: List[int] = context.application.bot_data["admin_ids"]

    await touch_user_profile(db, update.effective_user)

    if not await ensure_allowed(update, context, db, admin_ids):
        return

    msg = update.effective_message
    prompt = ((msg.text or "") if msg else "").strip()
    if not prompt:
        return

    file_ids: Optional[List[str]] = None

    # 1) reply текстом на фото/альбом — берём input из reply
    if msg and getattr(msg, "reply_to_message", None):
        rmsg = msg.reply_to_message
        chat_id = update.effective_chat.id

        ci = state.reply_inputs.get((chat_id, rmsg.message_id))
        if ci and (time.time() - ci.ts) <= CACHE_TTL_SEC:
            file_ids = ci.file_ids

        if not file_ids and getattr(rmsg, "media_group_id", None):
            key = (chat_id, str(rmsg.media_group_id))

            ci2 = state.media_group_inputs.get(key)
            if ci2 and (time.time() - ci2.ts) <= CACHE_TTL_SEC:
                file_ids = ci2.file_ids
            else:
                pending = state.pending_albums.get(key)
                if pending and pending.file_ids:
                    file_ids = list(pending.file_ids)

        if not file_ids:
            fid = pick_best_photo_file_id(rmsg)
            if fid:
                file_ids = [fid]

        if file_ids:
            state.cached.pop(update.effective_user.id, None)

    # 2) иначе — старое поведение: используем последний кеш фото/альбома
    if not file_ids:
        cached = state.cached.get(update.effective_user.id)
        if cached:
            if (time.time() - cached.ts) <= CACHE_TTL_SEC:
                file_ids = cached.file_ids
            state.cached.pop(update.effective_user.id, None)

    await enqueue_request(
        app=context.application,
        state=state,
        db=db,
        admin_ids=admin_ids,
        chat_id=update.effective_chat.id,
        user_id=update.effective_user.id,
        prompt=prompt,
        file_ids=file_ids,
        reply_to_message_id=(msg.message_id if msg else None),
    )


async def flush_album_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    state: BotState = app.bot_data["state"]
    db: Storage = app.bot_data["db"]
    admin_ids: List[int] = app.bot_data["admin_ids"]

    key = context.job.data["key"]
    pending: PendingAlbum = state.pending_albums.pop(key, None)
    state.album_jobs.pop(key, None)
    if not pending:
        return

    now = time.time()
    state.media_group_inputs[key] = CachedInputs(file_ids=pending.file_ids, ts=now)
    for mid in (pending.message_ids or []):
        state.reply_inputs[(pending.chat_id, mid)] = CachedInputs(file_ids=pending.file_ids, ts=now)

    if pending.caption and pending.caption.strip():
        await enqueue_request(
            app=app,
            state=state,
            db=db,
            admin_ids=admin_ids,
            chat_id=pending.chat_id,
            user_id=pending.user_id,
            prompt=pending.caption.strip(),
            file_ids=pending.file_ids,
            reply_to_message_id=pending.reply_to_message_id,
        )
        return

    state.cached[pending.user_id] = CachedInputs(file_ids=pending.file_ids, ts=time.time())
    await app.bot.send_message(
        chat_id=pending.chat_id,
        text="📥 Альбом получен. Теперь отправь текст-промпт, что сделать с этими фото.",
        reply_to_message_id=pending.reply_to_message_id,
    )


async def on_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: Storage = context.application.bot_data["db"]
    state: BotState = context.application.bot_data["state"]
    admin_ids: List[int] = context.application.bot_data["admin_ids"]

    await touch_user_profile(db, update.effective_user)

    if not await ensure_allowed(update, context, db, admin_ids):
        return

    msg = update.effective_message
    fid = pick_best_photo_file_id(msg)
    if not fid:
        return

    caption = (msg.caption or "").strip() if getattr(msg, "caption", None) else None

    if getattr(msg, "media_group_id", None):
        key = (update.effective_chat.id, str(msg.media_group_id))
        pending = state.pending_albums.get(key)
        if not pending:
            pending = PendingAlbum(
                chat_id=update.effective_chat.id,
                user_id=update.effective_user.id,
                file_ids=[],
                message_ids=[],
                caption=None,
                reply_to_message_id=msg.message_id,
                ts=time.time(),
            )
            state.pending_albums[key] = pending

        pending.file_ids.append(fid)
        pending.message_ids.append(msg.message_id)
        if caption:
            pending.caption = caption

        old_job = state.album_jobs.get(key)
        if old_job:
            old_job.schedule_removal()

        job = context.job_queue.run_once(
            flush_album_job,
            when=ALBUM_FLUSH_DELAY_SEC,
            data={"key": key},
            name=f"flush_album_{key[0]}_{key[1]}",
        )
        state.album_jobs[key] = job
        return

    state.reply_inputs[(update.effective_chat.id, msg.message_id)] = CachedInputs(file_ids=[fid], ts=time.time())

    if caption:
        await enqueue_request(
            app=context.application,
            state=state,
            db=db,
            admin_ids=admin_ids,
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            prompt=caption,
            file_ids=[fid],
            reply_to_message_id=msg.message_id,
        )
        return

    state.cached[update.effective_user.id] = CachedInputs(file_ids=[fid], ts=time.time())
    await msg.reply_text("📥 Фото получено. Теперь отправь текст-промпт, что сделать с этим изображением.")


# -----------------------------
# Cleanup job
# -----------------------------
async def cleanup_state_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    state: BotState = app.bot_data["state"]
    now = time.time()

    if state.cached:
        state.cached = {uid: ci for uid, ci in state.cached.items() if (now - ci.ts) <= CACHE_TTL_SEC}

    if state.pending_albums:
        max_album_age = 60.0
        state.pending_albums = {k: pa for k, pa in state.pending_albums.items() if (now - pa.ts) <= max_album_age}

    if state.cancelled:
        state.cancelled = {rid for rid in state.cancelled if rid in state.req_index}

    if state.reply_inputs:
        state.reply_inputs = {k: ci for k, ci in state.reply_inputs.items() if (now - ci.ts) <= CACHE_TTL_SEC}

    if state.media_group_inputs:
        state.media_group_inputs = {k: ci for k, ci in state.media_group_inputs.items() if (now - ci.ts) <= CACHE_TTL_SEC}


# -----------------------------
# Lifecycle
# -----------------------------
async def on_shutdown(app: Application) -> None:
    svcs: Dict[str, GeminiImageService] = app.bot_data.get("svcs") or {}
    for svc in svcs.values():
        try:
            await svc.aclose()
        except Exception:
            pass

    db: Storage = app.bot_data.get("db")
    if db:
        try:
            await db.aclose()
        except Exception:
            pass



def main() -> None:
    load_dotenv()

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    api_key = os.environ["GEMINI_API_KEY"]

    admin_ids = parse_ids_csv(os.getenv("ADMIN_IDS", ""))

    db_path = os.getenv("DB_PATH", "bot.sqlite3")
    default_ratio = os.getenv("DEFAULT_RATIO", "9:16")
    default_res = os.getenv("DEFAULT_RESOLUTION", "1K").upper()

    # глобальная модель по умолчанию (если не задано — Pro, как раньше)
    default_model = os.getenv("DEFAULT_MODEL_ID", MODEL_PRO_ID).strip()
    if default_model not in VALID_MODELS:
        default_model = MODEL_PRO_ID

    async def post_init(app: Application) -> None:
        db = Storage(db_path)
        await db.init()
        await db.seed_admins_as_allowed(admin_ids)

        try:
            if default_ratio in VALID_RATIOS:
                await db.set_global_ratio(default_ratio)
            if default_res in VALID_RESOLUTIONS:
                await db.set_global_resolution(default_res)
            if default_model in VALID_MODELS:
                await db.set_global_model(default_model)
        except Exception:
            pass

        # 2 сервиса под 2 модели
        svcs = {
            MODEL_FLASH_ID: GeminiImageService(api_key=api_key, model_id=MODEL_FLASH_ID),
            MODEL_PRO_ID: GeminiImageService(api_key=api_key, model_id=MODEL_PRO_ID),
        }

        app.bot_data["db"] = db
        app.bot_data["svcs"] = svcs
        app.bot_data["admin_ids"] = admin_ids
        app.bot_data["state"] = BotState()

        app.job_queue.run_repeating(
            cleanup_state_job,
            interval=CLEANUP_INTERVAL_SEC,
            first=CLEANUP_INTERVAL_SEC,
            name="cleanup_state",
        )

    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(on_shutdown)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("settings", cmd_settings))

    # Admin
    app.add_handler(CommandHandler("allow", cmd_allow))
    app.add_handler(CommandHandler("deny", cmd_deny))
    app.add_handler(CommandHandler("allowed", cmd_allowed))
    app.add_handler(CommandHandler("usage", cmd_usage))
    app.add_handler(CommandHandler("usage_month", cmd_usage_month))

    # Callbacks
    app.add_handler(CallbackQueryHandler(on_callback))

    # Images
    app.add_handler(MessageHandler(filters.PHOTO | (filters.Document.IMAGE), on_image))

    # Text (non-commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
