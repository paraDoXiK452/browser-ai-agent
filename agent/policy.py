"""Deterministic policy helpers for browser-agent runtime."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class EvaluationResult:
    status: str = "UNCLEAR"
    checkpoint_state: str = "PENDING"
    flags: list[str] = field(default_factory=list)
    evidence: str = ""
    correction: str = ""
    raw: str = ""

    @property
    def needs_fix(self) -> bool:
        return self.status in {"FIX", "UNCLEAR"}

    @property
    def is_complete(self) -> bool:
        return self.checkpoint_state == "COMPLETE"

    def has_flag(self, name: str) -> bool:
        return name in self.flags


@dataclass
class TaskProfile:
    kind: str
    domains: set[str]
    restaurant: str = ""
    requested_entities: list[str] = field(default_factory=list)

    @property
    def target_container(self) -> str:
        return self.restaurant

    @property
    def target_entities(self) -> list[str]:
        return self.requested_entities


_DELIVERY_HINTS = (
    "закаж",
    "достав",
    "корзин",
    "ресторан",
    "еда",
    "лавка",
    "delivery",
)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def infer_task_domains(text: str) -> set[str]:
    """Extract only explicitly mentioned domains from task text (no hardcoded mappings)."""
    normalized = normalize_text(text)
    domains: set[str] = set()
    for token in normalized.split():
        cleaned = token.strip(" ,;()[]<>\"'")
        if "." in cleaned and "://" not in cleaned and "/" not in cleaned:
            domains.add(cleaned)
    return domains


def extract_site_query(text: str) -> str:
    """Extract a concise service/site name from user task text for a web search.

    Parses patterns like "В яндекс еде закажи..." → "яндекс еда",
    "На hh.ru найди..." → "hh.ru".  Falls back to a truncated version of the task.
    """
    normalized = normalize_text(text)
    match = re.search(
        r"(?:^|\b)(?:в|на|через|из)\s+"
        r"([a-zа-яё][a-zа-яё0-9 .\-]{2,40}?)"
        r"(?:\s+(?:закаж|найд|откр|добав|удали|прочит|купи|оформ|отправ|зайд|посмотр|выбер|сдел))",
        normalized,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?:откр\w{0,6}|зайд\w{0,4}|перейд\w{0,4})\s+(?:в\s+|на\s+)?"
        r"([a-zа-яё][a-zа-яё0-9 .\-]{2,40}?)(?:\s|,|$)",
        normalized,
    )
    if match:
        return match.group(1).strip()
    return text.strip()[:80]


def infer_task_kind(text: str) -> str:
    normalized = normalize_text(text)
    if any(token in normalized for token in _DELIVERY_HINTS):
        return "delivery"
    if any(token in normalized for token in ("почта", "email", "mail", "спам", "письм")):
        return "mail"
    if any(token in normalized for token in ("ваканс", "hh", "headhunter", "рекрутер", "отклик")):
        return "jobs"
    return "generic"


def extract_requested_entities(task: str, *, task_kind: str) -> list[str]:
    normalized = normalize_text(task)
    entities: list[str] = []
    if task_kind == "delivery":
        patterns = [
            r"(?:закажи|заказать|добавь|добавить)\s+мне\s+(.+?)(?:\s+из\s+|\s+с\s+|,| но |$)",
            r"(?:закажи|заказать|добавь|добавить)\s+(.+?)(?:\s+из\s+|\s+с\s+|,| но |$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            chunk = match.group(1).strip(" ,.")
            parts = [part.strip(" ,.") for part in re.split(r"\s+и\s+|,", chunk) if part.strip(" ,.")]
            entities.extend(parts)
            break
    deduped: list[str] = []
    seen: set[str] = set()
    for item in entities:
        if len(item) < 3:
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def extract_target_restaurant(task: str, *, task_kind: str) -> str:
    normalized = normalize_text(task)
    if task_kind != "delivery":
        return ""
    patterns = [
        r"\b(?:с|из)\b\s+([a-zа-я0-9 .&\-—]+?)(?:,| но | на адрес| на указ| доставь| доставк| добавить| добавь| закаж| заказ| оформ|$)",
        r"\bресторан[а-я\s]*\s+([a-zа-я0-9 .&\-—]+?)(?:,| но | на адрес| на указ| доставь| доставк| добавить| добавь| закаж| заказ| оформ|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate = match.group(1).strip(" ,.\"'")
            candidate = re.sub(r"\s+", " ", candidate)
            if len(candidate) >= 3:
                return candidate
    return ""


def build_task_profile(task: str) -> TaskProfile:
    kind = infer_task_kind(task)
    return TaskProfile(
        kind=kind,
        domains=infer_task_domains(task),
        restaurant=extract_target_restaurant(task, task_kind=kind),
        requested_entities=extract_requested_entities(task, task_kind=kind),
    )


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zа-я0-9]+", normalize_text(text)) if token]


def _meaningful_tokens(text: str) -> list[str]:
    stop = {
        "мне", "и", "с", "из", "но", "в", "на", "по", "для", "или", "the", "a", "an",
        "big", "large",
    }
    return [token for token in _tokenize(text) if len(token) >= 4 and token not in stop]


def _entity_visible(entity: str, visible_text: str) -> bool:
    entity_tokens = _meaningful_tokens(entity)
    haystack_tokens = set(_tokenize(visible_text))
    if not entity_tokens:
        return False
    hits = 0
    for token in entity_tokens:
        if token in haystack_tokens:
            hits += 1
            continue
        if any(candidate.startswith(token[:5]) or token.startswith(candidate[:5]) for candidate in haystack_tokens if len(candidate) >= 5):
            hits += 1
    threshold = max(1, len(entity_tokens) - 1)
    return hits >= threshold


def infer_domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def infer_page_mode(*, current_url: str, page_text: str, flags: dict[str, bool] | None = None) -> str:
    flags = flags or {}
    normalized_url = current_url.lower()
    normalized_text = normalize_text(page_text)
    if flags.get("captcha") or "showcaptcha" in normalized_url or "я не робот" in normalized_text:
        return "captcha"
    if flags.get("address_modal"):
        return "address_gate"
    if "/cart" in normalized_url or "/checkout" in normalized_url or flags.get("cart_visible") or any(
        token in normalized_text for token in ("оформить заказ", "ваш заказ", "your order", "checkout")
    ):
        return "cart"
    return "unknown"


def body_fingerprint(body_text: str, max_len: int = 1600) -> str:
    """Compact fingerprint of visible text for detecting real page changes after search."""
    normalized = normalize_text(body_text or "")[:max_len]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:28]


def is_search_like_field(label: str, role: str) -> bool:
    """Heuristic: field is used for site search / filter / query (not address or single-line form name)."""
    r = normalize_text(role)
    l = normalize_text(label)
    if "searchbox" in r:
        return True
    if "combobox" in r and any(t in l for t in ("поиск", "найти", "search", "filter", "фильтр")):
        return True
    if "input" in r and "search" in l:
        return True
    markers = ("поиск", "найти", "search", "query", "filter", "фильтр", "искать")
    return any(m in l for m in markers)


def is_search_commit_control_click(label: str, role: str) -> bool:
    """Universal: primary search submit / apply controls (not site-specific routes)."""
    l = normalize_text(label)
    r = normalize_text(role)
    if r in {"option", "menuitem"}:
        return False
    keywords = (
        "найти",
        "искать",
        "search",
        "применить",
        "показать",
        "фильтровать",
        "go",
    )
    if any(k in l for k in keywords):
        return True
    if r in {"button", "a", "link", "img"} and any(k in l for k in ("поиск", "search", "найти")):
        return True
    return False


def is_address_like_field(label: str, role: str) -> bool:
    l = normalize_text(label)
    r = normalize_text(role)
    address_markers = (
        "адрес",
        "улица",
        "дом",
        "квартира",
        "подъезд",
        "доставить",
        "куда доставить",
        "введите адрес",
        "выбрать улицу",
    )
    return any(marker in l for marker in address_markers) and r not in {"button", "link", "a"}


def is_address_commit_control(label: str, role: str) -> bool:
    l = normalize_text(label)
    r = normalize_text(role)
    if r in {"option", "menuitem"}:
        return True
    commit_markers = (
        "выбрать",
        "подтвердить",
        "применить",
        "сохранить",
        "ок",
        "готово",
        "далее",
        "продолжить",
    )
    return any(marker in l for marker in commit_markers)


def is_search_suggestion_pick(role: str) -> bool:
    r = normalize_text(role)
    return r in {"option", "menuitem"}


def infer_search_scope(*, label: str, current_url: str, page_mode: str) -> str:
    lowered = normalize_text(label)
    search_markers = ("поиск", "найти", "search", "filter", "фильтр", "искать")
    if any(marker in lowered for marker in search_markers):
        return "search"
    return "unknown"


def is_probable_restaurant_card(label: str) -> bool:
    lowered = normalize_text(label)
    has_time_range = bool(re.search(r"\d+\s*[–—-]\s*\d+\s*мин", lowered))
    has_rating = bool(re.search(r"\d[.,]\d", lowered))
    generic_markers = ("доставка", "ресторан", "delivery", "restaurant")
    return (has_time_range and has_rating) or any(marker in lowered for marker in generic_markers)


def text_matches_target(label: str, target: str) -> bool:
    if not label or not target:
        return False
    label_tokens = set(_meaningful_tokens(label))
    target_tokens = set(_meaningful_tokens(target))
    if not target_tokens:
        return False
    overlap = len(label_tokens & target_tokens)
    if overlap >= max(1, len(target_tokens) - 1):
        return True
    label_norm = normalize_text(label)
    target_norm = normalize_text(target)
    return target_norm in label_norm or label_norm in target_norm


def is_authorization_request(question: str) -> bool:
    normalized = normalize_text(question)
    patterns = (
        "можно ",
        "можно ли",
        "разрешите",
        "подтвердите",
        "подтверждаете",
        "продолжить",
        "искать и открыть",
        "перейти к нужному ресторану",
        "добавить в корзину",
        "открыть",
    )
    return any(pattern in normalized for pattern in patterns)


def parse_evaluation(text: str) -> EvaluationResult:
    raw = text.strip()
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        # Fallback for malformed model output
        lower = cleaned.lower()
        flags: list[str] = []
        if "captcha" in lower or "recaptcha" in lower:
            flags.append("captcha")
        if any(token in lower for token in ["nothing found", "footer", "футер", "ничего не нашлось", "общий каталог"]):
            flags.append("dead_end")
        if "не тот" in lower or "wrong item" in lower:
            flags.append("wrong_item")
        if "не то поле" in lower or "wrong field" in lower:
            flags.append("wrong_field")
        return EvaluationResult(
            status="FIX" if "fix" in lower else ("OK" if "ok" in lower else "UNCLEAR"),
            checkpoint_state="COMPLETE" if "complete" in lower else ("DRIFTED" if "drifted" in lower else "PENDING"),
            flags=flags,
            evidence=cleaned[:280],
            correction=cleaned[:280],
            raw=raw,
        )

    status = str(data.get("status", "UNCLEAR")).upper()
    checkpoint_state = str(data.get("checkpoint_state", "PENDING")).upper()
    flags = [str(flag).lower() for flag in data.get("flags", []) if str(flag).strip()]
    evidence = str(data.get("evidence", "")).strip()
    correction = str(data.get("correction", "")).strip()
    if status not in {"OK", "FIX", "UNCLEAR"}:
        status = "UNCLEAR"
    if checkpoint_state not in {"COMPLETE", "PENDING", "DRIFTED"}:
        checkpoint_state = "PENDING"
    return EvaluationResult(
        status=status,
        checkpoint_state=checkpoint_state,
        flags=flags,
        evidence=evidence,
        correction=correction,
        raw=raw,
    )


def should_soft_accept_address(
    *,
    checkpoint_text: str,
    current_url: str,
    result: EvaluationResult,
) -> bool:
    checkpoint_lower = checkpoint_text.lower()
    if not any(token in checkpoint_lower for token in ["адрес", "достав", "получ"]):
        return False

    lower = result.raw.lower()
    hard_fail = any(token in lower for token in [
        "неверн", "другой адрес", "не распознан", "требуется уточнение",
        "адрес не найден", "не удалось определить",
    ])
    if hard_fail:
        return False
    if result.status == "OK" and result.checkpoint_state == "COMPLETE":
        return True
    address_indicators = ("адрес", "улиц", "достав", "дом ", "корпус", "подъезд", "address")
    has_address_context = any(ind in lower for ind in address_indicators)
    if result.status == "OK" and has_address_context:
        return True
    return False


def classify_dead_end(result: EvaluationResult) -> bool:
    if result.has_flag("dead_end"):
        return True
    lower = result.raw.lower()
    markers = [
        "ничего не нашлось",
        "футер",
        "footer",
        "низ страницы",
        "общий каталог",
        "общий экран",
        "пустой результат",
        "пустом результате",
    ]
    return any(marker in lower for marker in markers)


def task_uses_current_address(text: str) -> bool:
    normalized = normalize_text(text)
    markers = (
        "адрес который указан",
        "на адрес который указан",
        "на указанный адрес",
        "на текущий адрес",
        "на адрес из аккаунта",
        "на сохраненный адрес",
        "на сохранённый адрес",
        "адрес в профиле",
        "адрес уже указан",
    )
    return any(marker in normalized for marker in markers)


def task_has_explicit_address(text: str) -> bool:
    normalized = normalize_text(text)
    if task_uses_current_address(normalized):
        return False
    address_words = ("улиц", "ул ", "просп", "дом", "кварт", "подъезд", "корпус", "строен", "адрес")
    has_word = any(word in normalized for word in address_words)
    has_number = bool(re.search(r"\b\d+[a-zа-я/-]?\b", normalized))
    return has_word and has_number


def verify_task_completion(
    *,
    task: str,
    task_kind: str,
    current_url: str,
    page_text: str,
    current_checkpoint: str,
    result: EvaluationResult,
    has_uncommitted_search: bool = False,
) -> tuple[bool, str]:
    if has_uncommitted_search:
        return False, "search query was typed but not committed (submit Enter, search button, or pick a suggestion)"
    normalized_text = normalize_text(page_text)
    if result.status != "OK" or result.checkpoint_state != "COMPLETE":
        return False, "last evaluation is not complete"
    if result.has_flag("captcha"):
        return False, "captcha is still visible"
    if task_kind == "delivery":
        if not any(word in normalized_text for word in ("корзин", "оформ", "checkout")):
            return False, "cart or checkout state is not visible"
        if any(word in normalized_text for word in ("успешно оплачен", "заказ оформлен", "спасибо за заказ")):
            return False, "flow already passed the requested stop point"
        entities = extract_requested_entities(task, task_kind=task_kind)
        if entities:
            missing = [entity for entity in entities if not _entity_visible(entity, normalized_text)]
            if missing:
                return False, f"missing requested entities in visible state: {', '.join(missing[:3])}"
        if "не оплач" in normalize_text(task) and any(word in normalized_text for word in ("оплачено", "оплата прошла")):
            return False, "payment already happened"
    if not current_checkpoint:
        return True, "checkpoint-free verification passed"
    return True, "verification passed"
