# gemini_images.py
from __future__ import annotations

import asyncio
import base64
import logging
import random
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Iterable, List, Optional, Tuple

from PIL import Image
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    """Токены из usage_metadata (если API их вернул)."""
    prompt_tokens: int = 0
    # output tokens split
    image_output_tokens: int = 0
    text_output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


class ResponseMode(str, Enum):
    FILE = "file"
    PHOTO = "photo"


@dataclass
class UserSettings:
    response_mode: ResponseMode = ResponseMode.PHOTO


def update_settings_from_command(command: str, settings: UserSettings) -> Tuple[UserSettings, str]:
    """
    Update settings from a /settings command.
    Supported option: response_mode=<file|photo>
    """
    parts = command.strip().split(maxsplit=1)
    if len(parts) == 1:
        return settings, (
            "Settings:\n"
            "- response_mode=<file|photo> (send images as a file or as a photo)\n"
            f"Current response_mode={settings.response_mode.value}"
        )

    args = parts[1]
    key, _, value = args.partition("=")
    key = key.strip().lower()
    value = value.strip().lower()

    if key != "response_mode":
        return settings, "Unknown settings option. Use response_mode=<file|photo>."

    try:
        new_mode = ResponseMode(value)
    except ValueError:
        return settings, "Invalid response_mode. Use response_mode=<file|photo>."

    settings.response_mode = new_mode
    return settings, f"Updated response_mode={settings.response_mode.value}."


class GeminiImageService:
    """
    Notes:
    - You cannot fully disable safety. You CAN adjust per-category thresholds per request. :contentReference[oaicite:3]{index=3}
    - For image models, a blocked output may come back as FinishReason.IMAGE_SAFETY with no parts.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-3-pro-image-preview",
        # Recommended: "BLOCK_ONLY_HIGH" (aka "Block few") to reduce false positives without "disabling safety".
        # Other values: "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE", "OFF"
        # See docs for meanings. :contentReference[oaicite:4]{index=4}
        safety_threshold: str = "BLOCK_ONLY_HIGH",
        # Optional: pick API version if you need to align behavior (default SDK uses beta endpoints).
        http_api_version: Optional[str] = None,  # e.g. "v1alpha" for Gemini Developer API
    ):
        self._api_key = api_key
        self._model_id = model_id
        self._safety_threshold = safety_threshold

        http_options = None
        if http_api_version:
            http_options = types.HttpOptions(api_version=http_api_version)

        # async client
        self._client = genai.Client(api_key=api_key, http_options=http_options).aio

    @property
    def model_id(self) -> str:
        return self._model_id

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- helpers for robust attribute access ----------
    @staticmethod
    def _get(obj: Any, *names: str, default=None):
        for n in names:
            if obj is None:
                return default
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        return default

    @staticmethod
    def _trunc(s: str, n: int = 300) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 1] + "…"

    @staticmethod
    def _looks_like_png(data: bytes) -> bool:
        return isinstance(data, (bytes, bytearray)) and data[:8] == b"\x89PNG\r\n\x1a\n"

    @staticmethod
    def _ensure_png_bytes(data: bytes) -> bytes:
        """
        Не доверяем mime_type на 100% (бывает, что mime=png, а байты jpeg).
        Если это не PNG по сигнатуре — перекодируем через Pillow.
        """
        if GeminiImageService._looks_like_png(data):
            return data

        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")

        out = BytesIO()
        img.save(out, "PNG")
        return out.getvalue()

    @staticmethod
    def _iter_all_candidate_parts(resp) -> Iterable[object]:
        # 1) Основной путь: resp.candidates[*].content.parts[*]
        cands = getattr(resp, "candidates", None)
        if cands:
            for c in cands:
                content = getattr(c, "content", None)
                cparts = getattr(content, "parts", None) if content else None
                if cparts:
                    for p in cparts:
                        yield p
            # Если кандидаты есть — не дублируем через resp.parts
            return

        # 2) Фолбек: resp.parts
        parts = getattr(resp, "parts", None)
        if parts:
            for p in parts:
                yield p

    @staticmethod
    def _extract_images_from_parts(parts: Iterable[object]) -> List[bytes]:
        images: List[bytes] = []

        for part in parts:
            # 1) Удобный путь: part.as_image()
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                try:
                    img = as_image()
                    if isinstance(img, Image.Image):
                        out = BytesIO()
                        if img.mode not in ("RGB", "RGBA"):
                            img = img.convert("RGBA")
                        img.save(out, "PNG")
                        images.append(out.getvalue())
                        continue
                except Exception:
                    pass

            # 2) inline_data (snake_case) или inlineData (camelCase)
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if not inline:
                continue

            data = getattr(inline, "data", None)
            if not data:
                continue

            # data может быть bytes или base64-строкой
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data)
                except Exception:
                    continue

            if not isinstance(data, (bytes, bytearray)):
                continue

            images.append(GeminiImageService._ensure_png_bytes(bytes(data)))

        return images

    def _extract_usage(self, resp, has_image: bool) -> Usage:
        """
        Пытаемся достать usage metadata из ответа.
        В разных версиях SDK поля могут называться по-разному, поэтому делаем максимально робастно.
        """
        um = (
            getattr(resp, "usage_metadata", None)
            or getattr(resp, "usageMetadata", None)
            or getattr(resp, "usage", None)
        )
        if um is None:
            return Usage()

        # usage_metadata fields:
        # - prompt_token_count
        # - candidates_token_count (output, does NOT include thinking tokens)
        # - thoughts_token_count (thinking)
        # - total_token_count (prompt + candidates + thoughts)
        pt = self._get(um, "prompt_token_count", "promptTokenCount", default=0)
        ct = self._get(
            um,
            # older naming seen in some responses:
            "candidates_token_count",
            "candidatesTokenCount",
            "response_token_count",
            "responseTokenCount",
            default=0,
        )
        th = self._get(um, "thoughts_token_count", "thoughtsTokenCount", default=0)
        tt = self._get(um, "total_token_count", "totalTokenCount", default=0)

        if isinstance(um, dict):
            pt = um.get("promptTokenCount", um.get("prompt_token_count", pt or 0)) or 0
            ct = um.get(
                "candidatesTokenCount",
                um.get(
                    "candidates_token_count",
                    um.get("responseTokenCount", um.get("response_token_count", ct or 0)),
                ),
            ) or 0
            th = um.get("thoughtsTokenCount", um.get("thoughts_token_count", th or 0)) or 0
            tt = um.get("totalTokenCount", um.get("total_token_count", tt or 0)) or 0

        try:
            prompt = int(pt or 0)
            cand = int(ct or 0)
            thinking = int(th or 0)
            total = int(tt or 0)
            if has_image:
                return Usage(prompt_tokens=prompt, image_output_tokens=cand, text_output_tokens=0, thinking_tokens=thinking, total_tokens=total)
            return Usage(prompt_tokens=prompt, image_output_tokens=0, text_output_tokens=cand, thinking_tokens=thinking, total_tokens=total)
        except Exception:
            return Usage()

    def _no_image_debug_summary(self, resp) -> str:
        # prompt feedback (если промпт заблокировали)
        pf = getattr(resp, "prompt_feedback", None) or getattr(resp, "promptFeedback", None)
        block_reason = self._get(pf, "block_reason", "blockReason")
        pf_safety = self._get(pf, "safety_ratings", "safetyRatings")

        # candidates
        cands = getattr(resp, "candidates", None) or []
        cand_bits = []
        for i, c in enumerate(cands[:2], 1):
            fr = self._get(c, "finish_reason", "finishReason")
            sr = self._get(c, "safety_ratings", "safetyRatings")
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None

            part_kinds = []
            if parts:
                for p in parts[:8]:
                    if getattr(p, "text", None):
                        part_kinds.append("text")
                    inl = getattr(p, "inline_data", None) or getattr(p, "inlineData", None)
                    if inl:
                        mt = getattr(inl, "mime_type", None) or getattr(inl, "mimeType", None)
                        part_kinds.append(f"inline:{mt or '?'}")

            cand_bits.append(f"cand{i}(finish={fr}, safety={sr}, parts={part_kinds or '[]'})")

        # resp.text (коротко)
        text = getattr(resp, "text", None) or ""
        text = self._trunc(text, 400)

        bits = []
        if block_reason:
            bits.append(f"prompt_block={block_reason}")
        if pf_safety:
            bits.append(f"prompt_safety={pf_safety}")
        if cand_bits:
            bits.append("; ".join(cand_bits))
        if text:
            bits.append(f'text="{text}"')

        usage = self._extract_usage(resp, has_image=True)
        if (usage.total_tokens or usage.prompt_tokens or usage.image_output_tokens or usage.text_output_tokens or usage.thinking_tokens):
            bits.append(
                f"usage(prompt={usage.prompt_tokens}, img_out={usage.image_output_tokens}, text_out={usage.text_output_tokens}, thinking={usage.thinking_tokens}, total={usage.total_tokens})"
            )

        if not bits:
            bits.append("no extra fields in response")

        return " | ".join(bits)

    async def _call_with_retries(self, coro_factory, max_retries: int = 3) -> object:
        base_delay = 0.8
        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                msg = getattr(e, "message", "") or str(e)
                retryable = code in (429, 500, 502, 503, 504)
                if (attempt < max_retries) and retryable:
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.25)
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"Gemini API error {code}: {msg}") from e
            except Exception as e:
                raise RuntimeError(f"Gemini request failed: {e}") from e

        raise RuntimeError("Gemini request failed after retries")

    def _build_safety_settings(self) -> List[types.SafetySetting]:
        """
        Apply adjustable safety filters explicitly (helps align with AI Studio settings). :contentReference[oaicite:5]{index=5}
        """
        t = self._safety_threshold
        return [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=t),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=t),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=t),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=t),
        ]

    def _build_config(self, aspect_ratio: str, image_size: str) -> types.GenerateContentConfig:
        safety_settings = self._build_safety_settings()

        # IMPORTANT:
        # gemini-2.5-flash-image does NOT support image_size (fixed resolution per aspect_ratio)
        if self._model_id == "gemini-2.5-flash-image":
            return types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                safety_settings=safety_settings,
            )

        # For gemini-3-pro-image-preview we can request image (and optionally text).
        # If you only want images, switching to ["IMAGE"] can reduce surprises.
        return types.GenerateContentConfig(
            response_modalities=["IMAGE"],  # was ["TEXT","IMAGE"]
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
            safety_settings=safety_settings,
        )

    async def _generate_content(
        self,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        input_images: Optional[List[Image.Image]] = None,
    ) -> object:
        contents: List[object] = [prompt]
        if input_images:
            contents.extend(input_images)

        cfg = self._build_config(aspect_ratio=aspect_ratio, image_size=image_size)

        resp = await self._call_with_retries(
            lambda: self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=cfg,
            )
        )
        return resp

    # -----------------------------
    # Public API
    # -----------------------------
    async def generate_or_edit(
        self,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        input_images: Optional[List[Image.Image]] = None,
    ) -> List[bytes]:
        """
        Backward-compatible: returns ONLY images.
        If you need tokens — use generate_or_edit_with_usage().
        """
        images, _usage = await self.generate_or_edit_with_usage(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            input_images=input_images,
        )
        return images

    async def generate_or_edit_with_usage(
        self,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        input_images: Optional[List[Image.Image]] = None,
    ) -> Tuple[List[bytes], Usage]:
        """
        Returns (images, usage).
        usage can be zeros if SDK/model didn't return usage_metadata.
        """
        resp = await self._generate_content(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            input_images=input_images,
        )

        parts = list(self._iter_all_candidate_parts(resp))
        images = self._extract_images_from_parts(parts)
        usage = self._extract_usage(resp, has_image=True)

        if not images:
            summary = self._no_image_debug_summary(resp)
            logger.warning("No image returned by %s: %s", self._model_id, summary)

            # Helpful hint for your specific symptom
            # FinishReason.IMAGE_SAFETY typically means the image output was filtered.
            if "FinishReason.IMAGE_SAFETY" in summary or "IMAGE_SAFETY" in summary:
                hint = (
                    "Try another prompt | Попробуйте перефразировать запрос."
                )
                raise RuntimeError(f"Model returned no image. {summary} | {hint}")

            raise RuntimeError(f"Model returned no image. {summary}")

        return images, usage
