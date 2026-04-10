"""Compact execution memory for long-running browser tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.policy import body_fingerprint


@dataclass
class MemoryEvent:
    kind: str
    text: str


@dataclass
class TaskMemory:
    task: str
    summary: str = ""
    events: list[MemoryEvent] = field(default_factory=list)
    progress_note: str = ""
    last_evaluation: str = ""
    checkpoints: list[str] = field(default_factory=list)
    active_checkpoint_index: int = 0
    checkpoint_repeats: dict[str, int] = field(default_factory=dict)
    # Universal search discipline: after typing into a search-like field, commit before done/nav.
    pending_search_element_id: str = ""
    pending_search_text: str = ""
    pending_search_url_snapshot: str = ""
    pending_search_body_fp: str = ""
    pending_address_text: str = ""

    def add(self, kind: str, text: str) -> None:
        clean = " ".join(text.split())
        if clean:
            self.events.append(MemoryEvent(kind=kind, text=clean))

    def recent_events(self, limit: int = 14) -> str:
        if not self.events:
            return "Нет значимых событий."
        lines = [f"- [{event.kind}] {event.text}" for event in self.events[-limit:]]
        return "\n".join(lines)

    def task_context(self) -> str:
        parts = [f"Задача пользователя: {self.task}"]
        if self.checkpoints:
            parts.append(
                "Checkpoint plan:\n"
                + "\n".join(
                    f"{'->' if i == self.active_checkpoint_index else '  '} {i + 1}. {cp}"
                    for i, cp in enumerate(self.checkpoints)
                )
            )
        if self.progress_note:
            parts.append(f"Текущий фокус:\n{self.progress_note}")
        if self.summary:
            parts.append(f"Сжатая история:\n{self.summary}")
        hint = self.search_commit_hint()
        if hint:
            parts.append(hint)
        parts.append(f"Последние события:\n{self.recent_events()}")
        return "\n\n".join(parts)

    def should_compact(self, every_n_events: int = 16) -> bool:
        return len(self.events) >= every_n_events

    def replace_with_summary(self, summary: str) -> None:
        self.summary = summary.strip()
        self.events.clear()

    def update_progress(self, note: str) -> None:
        clean = " ".join(note.split())
        if clean:
            self.progress_note = clean

    def update_evaluation(self, note: str) -> None:
        clean = " ".join(note.split())
        if clean:
            self.last_evaluation = clean

    def set_checkpoints(self, checkpoints: list[str]) -> None:
        self.checkpoints = checkpoints[:]
        self.active_checkpoint_index = 0

    def current_checkpoint(self) -> str:
        if not self.checkpoints:
            return "Нет явного checkpoint."
        return self.checkpoints[min(self.active_checkpoint_index, len(self.checkpoints) - 1)]

    def advance_checkpoint(self) -> bool:
        if not self.checkpoints:
            return False
        if self.active_checkpoint_index < len(self.checkpoints) - 1:
            self.active_checkpoint_index += 1
            return True
        return False

    def note_checkpoint_repeat(self) -> int:
        current = self.current_checkpoint()
        count = self.checkpoint_repeats.get(current, 0) + 1
        self.checkpoint_repeats[current] = count
        return count

    def reset_checkpoint_repeat(self) -> None:
        self.checkpoint_repeats[self.current_checkpoint()] = 0

    def has_recent_progress(self, event_limit: int = 10) -> bool:
        progress_kinds = {
            "actions",
            "tool_result",
            "user",
            "candidate_done",
            "planner",
            "checkpoint",
        }
        return any(event.kind in progress_kinds for event in self.events[-event_limit:])

    def has_uncommitted_search(self) -> bool:
        return bool(self.pending_search_element_id.strip() and self.pending_search_text.strip())

    def search_commit_hint(self) -> str:
        if not self.has_uncommitted_search():
            return ""
        return (
            "Search commit (обязательно до done/навигации назад/внешнего поиска):\n"
            f"- В поле [{self.pending_search_element_id}] введён запрос «{self.pending_search_text}».\n"
            "- Зафиксируйте поиск: submit_observed_search(element_id) / Enter на этом поле, "
            "клик по кнопке поиска «Найти»/«Search», или выбор подходящего пункта в выпадашке.\n"
            "- Не очищайте поле и не вызывайте done(), пока запрос не отправлен или страница явно не сменилась."
        )

    def start_or_update_search_commit(
        self,
        *,
        element_id: str,
        query: str,
        url: str,
        body_text: str,
    ) -> None:
        clean_q = " ".join(query.split()).strip()
        if not clean_q:
            return
        self.pending_search_element_id = element_id.strip()
        self.pending_search_text = clean_q
        self.pending_search_url_snapshot = (url or "").strip()
        self.pending_search_body_fp = body_fingerprint(body_text or "")
        self.add(
            "search_session",
            f"Запрос «{clean_q}» в поле {element_id}; нужно зафиксировать поиск (Enter / кнопка / пункт списка).",
        )

    def end_search_commit(self) -> None:
        if self.pending_search_element_id:
            self.add("search_session", "Поиск зафиксирован (отправка или смена страницы).")
        self.pending_search_element_id = ""
        self.pending_search_text = ""
        self.pending_search_url_snapshot = ""
        self.pending_search_body_fp = ""

    def has_pending_address(self) -> bool:
        return bool(self.pending_address_text.strip())

    def start_pending_address(self, text: str) -> None:
        clean = " ".join(text.split()).strip()
        if not clean:
            return
        self.pending_address_text = clean
        self.add("address_session", f"Введен адрес «{clean}»; нужно выбрать или подтвердить результат.")

    def end_pending_address(self) -> None:
        if self.pending_address_text:
            self.add("address_session", "Адрес подтвержден или выбран из результатов.")
        self.pending_address_text = ""
