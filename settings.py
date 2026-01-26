from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple


class SendResultMode(str, Enum):
    FILE = "file"
    PHOTO = "photo"

    def label(self) -> str:
        return "File" if self is SendResultMode.FILE else "Photo"


@dataclass(frozen=True)
class UserSettings:
    send_result_mode: SendResultMode = SendResultMode.PHOTO

    def describe(self) -> str:
        return f"Send result as: {self.send_result_mode.label()}"


class SettingsStore:
    """In-memory settings registry for /settings handlers."""

    def __init__(self) -> None:
        self._settings: Dict[int, UserSettings] = {}

    def get(self, user_id: int) -> UserSettings:
        return self._settings.get(user_id, UserSettings())

    def set_send_result_mode(self, user_id: int, mode: SendResultMode) -> UserSettings:
        settings = self.get(user_id)
        if settings.send_result_mode is mode:
            return settings
        updated = UserSettings(send_result_mode=mode)
        self._settings[user_id] = updated
        return updated

    @staticmethod
    def settings_options() -> Iterable[Tuple[str, SendResultMode]]:
        return [
            ("Send as file", SendResultMode.FILE),
            ("Send as photo", SendResultMode.PHOTO),
        ]

    def build_settings_message(self, user_id: int) -> str:
        settings = self.get(user_id)
        lines = ["Settings:", settings.describe(), "", "Choose how to send results:"]
        return "\n".join(lines)
