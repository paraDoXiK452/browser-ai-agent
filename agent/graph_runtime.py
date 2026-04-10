"""LangGraph-based browser agent runtime."""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent.browser_tools import BrowserConfig, BrowserSession
from agent.llm import LLMClient
from agent.memory import TaskMemory
from agent.policy import (
    EvaluationResult,
    TaskProfile,
    build_task_profile,
    classify_dead_end,
    extract_requested_entities,
    extract_site_query,
    infer_domain_from_url,
    infer_task_domains,
    infer_task_kind,
    infer_page_mode,
    infer_search_scope,
    is_address_commit_control,
    is_address_like_field,
    is_authorization_request,
    is_probable_restaurant_card,
    is_search_commit_control_click,
    is_search_like_field,
    is_search_suggestion_pick,
    parse_evaluation,
    should_soft_accept_address,
    task_has_explicit_address,
    task_uses_current_address,
    text_matches_target,
    verify_task_completion,
)
from agent.prompts import get_executor_prompt
from agent.subagents import CheckpointPlanner, HistorySummarizer, TaskEvaluator, TaskPlanner


class AgentState(TypedDict, total=False):
    task: str
    step: int
    step_limit: int
    consecutive_stalls: int
    response: Any
    response_id: str
    computer_call: Any
    function_calls: list[Any]
    text_parts: list[str]
    recent_actions: list[str]
    screenshot_url: str
    evaluation: EvaluationResult
    done_verified: bool
    completed_successfully: bool
    finished_message: str
    stop_reason: str
    include_planner: bool


@dataclass
class AgentDeps:
    console: Console
    browser: BrowserSession
    llm: LLMClient
    planner: TaskPlanner
    checkpoint_planner: CheckpointPlanner
    summarizer: HistorySummarizer
    evaluator: TaskEvaluator
    memory: TaskMemory
    soft_max: int
    hard_max: int
    extend_by: int
    compact_every: int
    keep_browser_open: bool
    startup_timeout: int
    allow_external_search: bool
    trace_path: Path
    task_domains: set[str]
    task_kind: str
    requested_entities: list[str]
    profile: TaskProfile

    def log_event(self, kind: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            **payload,
        }
        try:
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def _fallback_plan() -> str:
    return (
        "1. Goal\nПродолжить выполнение задачи по текущему экрану.\n\n"
        "2. Next steps\nПроверить текущее состояние страницы и сделать следующий осмысленный шаг.\n\n"
        "3. Risks\nНе переходить к оплате и не выполнять деструктивные действия без явной необходимости.\n\n"
        "4. Completion signals\nИтоговое состояние на экране соответствует запросу пользователя."
    )


def _render_eval(r: EvaluationResult) -> str:
    flags = ", ".join(r.flags) if r.flags else "-"
    return f"Status: {r.status}\nCheckpoint: {r.checkpoint_state}\nFlags: {flags}\n\nEvidence: {r.evidence or '-'}\n\nCorrection: {r.correction or '-'}"


def _normalize_domain(url: str) -> str:
    return infer_domain_from_url(url)


def _extract_domains(text: str) -> set[str]:
    domains = set()
    for token in text.replace("\n", " ").split():
        token = token.strip(" ,;()[]<>\"'")
        if "." in token and "://" not in token and "/" not in token:
            domains.add(token.lower())
    return domains


def _current_checkpoint_is_last(memory: TaskMemory) -> bool:
    return bool(memory.checkpoints) and memory.active_checkpoint_index >= len(memory.checkpoints) - 1


def _done_gate_satisfied(memory: TaskMemory) -> bool:
    if not memory.checkpoints or not _current_checkpoint_is_last(memory):
        return False
    if not memory.last_evaluation:
        return False
    result = parse_evaluation(memory.last_evaluation)
    if result.status != "OK" or result.checkpoint_state != "COMPLETE":
        return False
    return result.has_flag("ready_to_finish")


def _last_eval_signals_finish(memory: TaskMemory) -> bool:
    """True if the latest evaluator output already marks the task as ready to complete."""
    if not memory.last_evaluation:
        return False
    result = parse_evaluation(memory.last_evaluation)
    return result.status == "OK" and result.checkpoint_state == "COMPLETE" and result.has_flag("ready_to_finish")


def _parse_page_state(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except Exception:
        return {"url": "", "title": "", "body_text": "", "dialog_texts": [], "overlay_texts": [], "flags": {}}
    data.setdefault("flags", {})
    data.setdefault("dialog_texts", [])
    data.setdefault("overlay_texts", [])
    data.setdefault("body_text", "")
    data.setdefault("title", "")
    data.setdefault("url", "")
    return data


def _page_state_summary(page_state: dict[str, Any]) -> str:
    flags = page_state.get("flags", {})
    page_mode = infer_page_mode(
        current_url=str(page_state.get("url", "")),
        page_text=str(page_state.get("body_text", "")),
        flags=flags,
    )
    active_flags = [name for name, value in flags.items() if value]
    snippets = []
    for item in list(page_state.get("dialog_texts", []))[:2] + list(page_state.get("overlay_texts", []))[:2]:
        if item:
            snippets.append(item[:180])
    parts = [
        f"Current URL: {page_state.get('url', '-') or '-'}",
        f"Page title: {page_state.get('title', '-') or '-'}",
        f"Page mode: {page_mode}",
        f"Detected flags: {', '.join(active_flags) if active_flags else '-'}",
    ]
    if snippets:
        parts.append("Visible blocker/context snippets:\n" + "\n".join(f"- {item}" for item in snippets))
    body_text = str(page_state.get("body_text", "")).strip()
    if body_text:
        parts.append(f"Visible page text excerpt:\n{body_text[:600]}")
    return "\n".join(parts)


def _build_final_message(deps: AgentDeps, page_state: dict[str, Any]) -> str:
    parts: list[str] = []
    if deps.profile.target_entities:
        parts.append(", ".join(deps.profile.target_entities))
    elif deps.profile.requested_entities:
        parts.append(", ".join(deps.profile.requested_entities))
    elif deps.profile.target_container:
        parts.append(deps.profile.target_container)
    item_text = parts[0] if parts else "нужный результат"
    if deps.task_kind == "delivery":
        msg = f"В корзине: {item_text}."
        if deps.profile.target_container:
            msg = f"В корзине: {item_text} из {deps.profile.target_container}."
        msg += " До оплаты я не переходил."
        return msg
    return f"Задача доведена до видимого финального состояния: {item_text}."


def _normalize_item_label(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").split())


def _expected_delivery_items(profile: TaskProfile) -> list[tuple[str, int]]:
    entities = profile.target_entities or profile.requested_entities
    result: list[tuple[str, int]] = []
    for entity in entities:
        clean = _normalize_item_label(entity)
        if clean:
            result.append((clean, 1))
    return result


def _cart_exact_match(profile: TaskProfile, cart: dict[str, Any]) -> tuple[bool, str]:
    """Ensure each requested entity is present with enough qty. Extra snapshot lines (menu noise) are ignored."""
    expected = _expected_delivery_items(profile)
    items = cart.get("items", []) if isinstance(cart.get("items", []), list) else []
    actual: list[tuple[str, int]] = []
    for it in items:
        name = _normalize_item_label(it.get("name", ""))
        if not name:
            continue
        qty = it.get("qty", 1)
        try:
            qty_int = int(qty)
        except Exception:
            qty_int = 1
        actual.append((name, max(1, qty_int)))
    if not expected:
        return True, "no explicit requested entities"
    if not actual:
        return False, "cart snapshot has no parsed item lines"
    remaining: list[tuple[str, int]] = list(actual)
    for expected_name, expected_qty in expected:
        need = expected_qty
        i = 0
        while need > 0 and i < len(remaining):
            aname, aqty = remaining[i]
            if text_matches_target(aname, expected_name):
                take = min(need, aqty)
                need -= take
                new_qty = aqty - take
                if new_qty <= 0:
                    remaining.pop(i)
                else:
                    remaining[i] = (aname, new_qty)
                    i += 1
            else:
                i += 1
        if need > 0:
            return False, f"missing requested item in cart snapshot: {expected_name} x{expected_qty}"
    return True, "all requested items matched in cart snapshot"


def _summarize_observation(raw: str, limit: int = 12) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return ""
    url = str(data.get("url", "")).strip()
    goal = str(data.get("goal", "")).strip()
    elements = data.get("elements", [])
    lines: list[str] = []
    for item in elements[:limit]:
        label = str(item.get("label", "")).strip() or "(no label)"
        role = str(item.get("role", "")).strip() or "element"
        element_id = str(item.get("id", "")).strip()
        lines.append(f"- [{element_id}] {role}: {label}")
    if not lines:
        return f"Live observation:\nURL: {url or '-'}"
    header = "Live observation:"
    if url:
        header += f"\nURL: {url}"
    if goal:
        header += f"\nGoal hint: {goal}"
    return f"{header}\n" + "\n".join(lines)


async def _build_graph(deps: AgentDeps):
    graph = StateGraph(AgentState)

    async def bootstrap(state: AgentState) -> AgentState:
        deps.console.print(f"  [dim]Model: {deps.llm.model}[/dim]")
        deps.console.print(f"  [dim]Trace: {deps.trace_path}[/dim]")
        deps.log_event("run_started", task=state["task"], model=deps.llm.model)
        if deps.profile.target_container or deps.profile.target_entities:
            deps.memory.update_progress(
                "Task profile:\n"
                f"- kind: {deps.profile.kind}\n"
                f"- target container: {deps.profile.target_container or '-'}\n"
                f"- target entities: {', '.join(deps.profile.target_entities) if deps.profile.target_entities else '-'}"
            )
        await deps.browser.start()
        deps.console.print(Panel("[bold green]Browser opened[/bold green]", title="Agent"))
        deps.console.print("  [dim]Building checkpoints...[/dim]")
        try:
            checkpoints = await asyncio.wait_for(deps.checkpoint_planner.build(state["task"]), timeout=deps.startup_timeout)
        except Exception as exc:
            checkpoints = []
            deps.console.print(f"  [yellow]Checkpoint planner fallback: {exc if exc else 'timeout'}[/yellow]")
        if checkpoints:
            deps.memory.set_checkpoints(checkpoints)
            deps.log_event("checkpoints_built", checkpoints=checkpoints)
            deps.console.print(Panel("\n".join(f"- {cp}" for cp in checkpoints), title="Checkpoints", border_style="green"))
        return {"step": 0, "step_limit": deps.soft_max, "consecutive_stalls": 0, "done_verified": False, "include_planner": True}

    async def plan(state: AgentState) -> AgentState:
        if not state.get("include_planner", True):
            return {}
        deps.console.print("  [dim]Planning next steps...[/dim]")
        try:
            notes = await asyncio.wait_for(deps.planner.plan(deps.memory.task_context()), timeout=deps.startup_timeout)
        except Exception as exc:
            notes = _fallback_plan()
            deps.console.print(f"  [yellow]Planner fallback: {exc if exc else 'timeout'}[/yellow]")
        deps.memory.add("planner", notes)
        deps.memory.update_progress(notes)
        deps.log_event("planner_notes", notes=notes)
        deps.console.print(Panel(notes, title="Planner", border_style="blue"))
        return {"include_planner": False}

    async def ensure_start_page(state: AgentState) -> AgentState:
        current_url = (await deps.browser.current_url()).strip()
        blank_urls = {"", "about:blank", "chrome://newtab/", "chrome://new-tab-page/"}
        if current_url not in blank_urls:
            deps.log_event("start_page_ready", url=current_url)
            return {}

        deps.console.print("  [dim]Opening initial page...[/dim]")
        if deps.task_domains:
            for domain in sorted(deps.task_domains):
                target = f"https://{domain}"
                result = await deps.browser.navigate(target)
                deps.memory.add("tool_result", result)
                deps.log_event("bootstrap_navigate", url=target, result=result)
                deps.console.print(f"  [green]-> {result[:120]}[/green]")
                if "Navigation failed" not in result:
                    return {}

        query = extract_site_query(state["task"])
        result = await deps.browser.search_web(query)
        deps.memory.add("tool_result", result)
        deps.log_event("bootstrap_search", query=query, result=result)
        deps.console.print(f"  [green]-> {result[:120]}[/green]")
        return {}

    async def start_executor(state: AgentState) -> AgentState:
        deps.console.print("  [dim]Starting executor...[/dim]")
        screenshot = await deps.browser.screenshot()
        observation = ""
        page_state: dict[str, Any] = {}
        try:
            page_state = _parse_page_state(await deps.browser.inspect_state())
            deps.memory.update_progress(_page_state_summary(page_state))
            deps.log_event("page_state", page_state=page_state)
        except Exception as exc:
            deps.log_event("page_state_error", error=str(exc))
        try:
            observation_raw = await deps.browser.observe(deps.memory.current_checkpoint() or state["task"])
            observation = _summarize_observation(observation_raw)
            if observation:
                deps.log_event("live_observation", observation=observation)
        except Exception as exc:
            deps.log_event("live_observation_error", error=str(exc))
        task_context = deps.memory.task_context()
        if page_state:
            task_context = f"{task_context}\n\nDeterministic page state:\n{_page_state_summary(page_state)}"
        if observation:
            task_context = f"{task_context}\n\n{observation}"
        response = await asyncio.wait_for(
            deps.llm.restart_from_context(get_executor_prompt(task_context), task_context, screenshot),
            timeout=90.0,
        )
        return {"response": response, "response_id": response.id, "screenshot_url": screenshot}

    async def classify(state: AgentState) -> AgentState:
        response = state["response"]
        computer_call = None
        function_calls: list[Any] = []
        text_parts: list[str] = []
        for item in response.output:
            if item.type == "computer_call":
                computer_call = item
            elif item.type == "function_call":
                function_calls.append(item)
            elif item.type == "message":
                for block in item.content:
                    if hasattr(block, "text") and block.text:
                        text_parts.append(block.text)
        if text_parts:
            text = " ".join(text_parts)[:300]
            deps.console.print(f"  [dim italic]{text}[/dim italic]")
            deps.memory.add("assistant", text)
            deps.log_event("assistant_text", text=text)
        return {"computer_call": computer_call, "function_calls": function_calls, "text_parts": text_parts}

    async def handle_computer(state: AgentState) -> AgentState:
        call = state["computer_call"]
        actions = call.actions
        step = state["step"] + 1
        action_descs = []
        for a in actions:
            if a.type == "click":
                action_descs.append(f"click({a.x}, {a.y})")
            elif a.type == "type":
                action_descs.append(f"type({a.text[:40]!r})")
            elif a.type == "keypress":
                action_descs.append(f"keypress({a.keys})")
            elif a.type == "scroll":
                action_descs.append(f"scroll({getattr(a, 'scroll_y', 0)})")
            else:
                action_descs.append(a.type)
        deps.console.print(f"  [bold]Step {step}/{state['step_limit']}[/bold] [cyan]{', '.join(action_descs)}[/cyan]")
        deps.memory.add("actions", ", ".join(action_descs))
        deps.log_event("computer_actions", step=step, actions=action_descs)

        approved = True
        safety_checks = getattr(call, "pending_safety_checks", None)
        if safety_checks:
            lines = []
            for check in safety_checks:
                message = getattr(check, "message", str(check))
                code = getattr(check, "code", "")
                lines.append(f"- {code}: {message}" if code else f"- {message}")
            joined = " ".join(lines).lower()
            risky = any(m in joined for m in ["payment", "pay", "checkout", "delete", "password", "security", "оплат", "удал", "парол", "безопас"])
            allowed = any(m in state["task"].lower() for m in ["закажи", "добавь в корзину", "отправь", "submit", "send", "add to cart"])
            if risky or not allowed:
                deps.console.print(Panel("Модель пытается выполнить потенциально рискованное действие.\n" + "\n".join(lines) + "\n\nПодтвердить выполнение?", title="Safety Approval", border_style="yellow"))
                approved = Prompt.ask("[bold]Введите yes/no[/bold]", default="no").strip().lower() in {"y", "yes", "да", "lf"}
        current_domain = _normalize_domain(await deps.browser.current_url())
        forbidden = {("CTRL", "L"), ("CONTROL", "L"), ("ALT", "D"), ("CTRL", "K"), ("CONTROL", "K"), ("CTRL", "F"), ("CONTROL", "F")}
        for a in actions:
            if current_domain and current_domain not in {"google.com", "bing.com"} and getattr(a, "type", "") == "keypress":
                keys = tuple(str(k).upper() for k in getattr(a, "keys", []))
                if keys in forbidden:
                    approved = False
                    deps.console.print("  [yellow]-> REJECTED. Do not use browser shortcuts like Ctrl+L, Ctrl+F, Alt+D, or Ctrl+K inside an active web app.[/yellow]")
            if current_domain and current_domain not in {"google.com", "bing.com"} and getattr(a, "type", "") == "type":
                text = str(getattr(a, "text", "")).lower()
                if text.startswith(("http://", "https://", "www.")) or ".ru" in text or ".com" in text:
                    approved = False
                    deps.console.print("  [yellow]-> REJECTED. Do not type raw URLs or domains into ordinary page fields inside an active web app.[/yellow]")

        if approved:
            try:
                await asyncio.wait_for(deps.browser.execute_actions(actions), timeout=30.0)
            except Exception as exc:
                deps.memory.add("error", f"Action error: {exc}")
                deps.log_event("action_error", step=step, error=str(exc))
                deps.console.print(f"  [red]Action error (continuing): {exc}[/red]")

        await deps.browser.close_extra_tabs()
        await asyncio.sleep(1.0)
        screenshot_url = await deps.browser.screenshot()
        mixed_outputs: list[dict[str, Any]] = [{
            "type": "computer_call_output",
            "call_id": call.call_id,
            "output": {"type": "computer_screenshot", "image_url": screenshot_url, "detail": "original"},
        }]

        function_calls = state.get("function_calls", [])
        completed_successfully = False
        finished_message = ""
        done_verified = state.get("done_verified", False)
        for fc in function_calls:
            should_return, finished_message, done_verified, output = await _process_function_call(fc, state, deps, done_verified, step)
            mixed_outputs.append({"type": "function_call_output", "call_id": fc.call_id, "output": output})
            if should_return:
                completed_successfully = True
                break

        if completed_successfully:
            return {"step": step, "completed_successfully": True, "finished_message": finished_message, "done_verified": done_verified}

        response = await deps.llm.send_tool_outputs(state["response_id"], mixed_outputs)
        return {
            "step": step,
            "response": response,
            "response_id": response.id,
            "screenshot_url": screenshot_url,
            "recent_actions": action_descs,
            "done_verified": done_verified,
            "consecutive_stalls": 0,
        }

    async def evaluate(state: AgentState) -> AgentState:
        result = await deps.evaluator.evaluate(
            deps.memory.task_context(),
            deps.memory.current_checkpoint(),
            ", ".join(state.get("recent_actions", [])),
            state["screenshot_url"],
        )
        deps.memory.add("evaluation", result.raw)
        deps.memory.update_evaluation(result.raw)
        deps.log_event("evaluation", checkpoint=deps.memory.current_checkpoint(), status=result.status, checkpoint_state=result.checkpoint_state, flags=result.flags, evidence=result.evidence, correction=result.correction)
        deps.console.print(Panel(_render_eval(result), title="Evaluator", border_style="magenta"))
        return {"evaluation": result}

    async def handle_functions(state: AgentState) -> AgentState:
        step = state["step"] + 1
        outputs: list[dict[str, Any]] = []
        done_verified = state.get("done_verified", False)
        for fc in state.get("function_calls", []):
            should_return, finished_message, done_verified, output = await _process_function_call(fc, state, deps, done_verified, step)
            outputs.append({"call_id": fc.call_id, "output": output})
            if should_return:
                return {"step": step, "completed_successfully": True, "finished_message": finished_message, "done_verified": done_verified}
        response = await deps.llm.send_function_outputs(state["response_id"], outputs)
        return {"step": step, "response": response, "response_id": response.id, "done_verified": done_verified, "consecutive_stalls": 0}

    async def nudge(state: AgentState) -> AgentState:
        stalls = state.get("consecutive_stalls", 0) + 1
        deps.console.print(f"  [dim]No actions, nudging model to continue (stall #{stalls})...[/dim]")
        deps.memory.add("stall", f"stall #{stalls}")
        deps.log_event("stall", count=stalls)
        response = await deps.llm.nudge(state["response_id"], severity=stalls)
        return {"response": response, "response_id": response.id, "consecutive_stalls": stalls, "step": state["step"] + 1}

    async def recover(state: AgentState) -> AgentState:
        result = state["evaluation"]
        current_url = await deps.browser.current_url()
        page_state: dict[str, Any] = {}
        try:
            page_state = _parse_page_state(await deps.browser.inspect_state())
            deps.log_event("recovery_page_state", page_state=page_state)
        except Exception as exc:
            deps.log_event("recovery_page_state_error", error=str(exc))
        if result.has_flag("captcha"):
            deps.console.print(Panel("На странице видна CAPTCHA. Реши её вручную в браузере, затем нажми Enter здесь.", title="Manual Action Required", border_style="yellow"))
            Prompt.ask("[bold]Нажми Enter после решения CAPTCHA[/bold]", default="")
            try:
                pass
            except EOFError:
                return {"stop_reason": "Manual CAPTCHA action required, but stdin is unavailable."}
            deps.memory.add("user", "Solved CAPTCHA manually")
            deps.log_event("manual_captcha_solved")
            return {"include_planner": False}
        if page_state.get("flags", {}).get("cookie_banner") or page_state.get("flags", {}).get("modal_visible"):
            dismiss_result = await deps.browser.dismiss_blockers()
            deps.memory.add("tool_result", dismiss_result)
            deps.log_event("recovery_dismiss_blocker", result=dismiss_result)
            if "Dismissed blocker" in dismiss_result:
                deps.memory.update_progress(
                    "A visible blocker was dismissed. Continue from the same local app context before using back navigation."
                )
                return {"include_planner": False}
        if classify_dead_end(result):
            note = f"Текущий checkpoint: {deps.memory.current_checkpoint()}. Ты попал в тупиковое состояние. Вернись к рабочему контенту через back/close/scroll и продолжай через локальный UI."
            deps.memory.update_progress(note)
            deps.log_event("dead_end_recovery", checkpoint=deps.memory.current_checkpoint(), note=note)
        else:
            deps.memory.update_progress(result.correction or result.evidence or result.raw)
        if result.has_flag("wrong_destination") or result.has_flag("wrong_item") or result.has_flag("wrong_search_context"):
            deps.memory.update_progress(
                "Recovery policy: first try local correction inside the current app. "
                "Close overlays, clear search context, or reopen the correct menu before using go_back(). "
                "Then use observe() to identify the exact visible restaurant, product, or field before clicking or typing."
            )
        if (
            task_uses_current_address(deps.memory.task)
            and result.has_flag("wrong_destination")
            and not task_has_explicit_address(deps.memory.task)
        ):
            body_text = str(page_state.get("body_text", "")).lower()
            if any(token in body_text for token in ("улиц", "адрес", "достав", "address", "delivery")):
                deps.memory.update_progress(
                    "The task refers to the currently selected account address rather than a newly specified exact address. "
                    "Do not require a new address entry unless the site explicitly blocks progress."
                )
                if deps.memory.advance_checkpoint():
                    deps.memory.add("checkpoint", f"advanced to: {deps.memory.current_checkpoint()}")
                return {"include_planner": False}
        current_page_mode = infer_page_mode(
            current_url=page_state.get("url", ""),
            page_text=page_state.get("body_text", ""),
            flags=page_state.get("flags", {}),
        )
        should_go_back = classify_dead_end(result) or (
            (result.has_flag("wrong_destination") or result.has_flag("wrong_search_context"))
            and not page_state.get("flags", {}).get("modal_visible")
            and not page_state.get("flags", {}).get("address_modal")
            and current_page_mode not in {"address_gate", "captcha", "cart"}
        )
        if should_go_back:
            back_result = await deps.browser.go_back()
            deps.memory.add("tool_result", back_result)
            deps.log_event("recovery_go_back", result=back_result)
        if should_soft_accept_address(checkpoint_text=deps.memory.current_checkpoint(), current_url=current_url, result=result):
            deps.memory.add("checkpoint", f"soft-accepted: {deps.memory.current_checkpoint()}")
            deps.memory.reset_checkpoint_repeat()
            if deps.memory.advance_checkpoint():
                deps.memory.add("checkpoint", f"advanced to: {deps.memory.current_checkpoint()}")
        else:
            repeat_count = deps.memory.note_checkpoint_repeat()
            deps.log_event("checkpoint_repeat", checkpoint=deps.memory.current_checkpoint(), count=repeat_count)
        if deps.memory.should_compact(deps.compact_every):
            summary = await deps.summarizer.summarize(deps.memory.task_context())
            deps.memory.replace_with_summary(summary)
            deps.log_event("context_compacted", reason="recover", summary=summary)
            deps.console.print(Panel(summary, title="Memory Summary", border_style="cyan"))
        return {"include_planner": False}

    async def finish(state: AgentState) -> AgentState:
        return state

    def route_from_classify(state: AgentState) -> str:
        if state.get("completed_successfully"):
            return "finish"
        if state.get("computer_call") is not None:
            return "handle_computer"
        if state.get("function_calls"):
            return "handle_functions"
        # Model produced no actions while evaluator already marked success — avoid burning steps/LLM $.
        stall_cap = max(1, int(os.getenv("STALL_FORCE_FINISH", "2").strip() or "2"))
        if state.get("consecutive_stalls", 0) >= stall_cap and _last_eval_signals_finish(deps.memory):
            deps.console.print(
                f"  [dim]Stall breaker: forcing done (stall ≥{stall_cap}, evaluator already ready_to_finish).[/dim]"
            )
            deps.log_event("stall_force_finalize", stalls=state.get("consecutive_stalls", 0))
            return "finalize_success"
        return "nudge"

    def route_after_computer(state: AgentState) -> str:
        if state.get("completed_successfully"):
            return "finish"
        if state["step"] >= state["step_limit"]:
            if state["step_limit"] < deps.hard_max and deps.memory.has_recent_progress():
                state["step_limit"] = min(state["step_limit"] + deps.extend_by, deps.hard_max)
                deps.console.print(f"  [yellow]Soft step limit reached, extending budget to {state['step_limit']} due to recent progress.[/yellow]")
                deps.memory.add("limit", f"extended to {state['step_limit']}")
                return "evaluate"
            state["stop_reason"] = "step limit reached"
            return "finish"
        return "evaluate"

    def route_after_evaluate(state: AgentState) -> str:
        if state.get("completed_successfully"):
            return "finish"
        result = state["evaluation"]
        if result.is_complete:
            deps.memory.reset_checkpoint_repeat()
            # As soon as the evaluator says we're done, finalize — do not advance through
            # remaining checkpoint labels (wastes steps and often causes stall loops).
            if result.status == "OK" and result.has_flag("ready_to_finish"):
                deps.log_event("route_finalize", reason="evaluator_ready_to_finish")
                return "finalize_success"
            if deps.memory.advance_checkpoint():
                deps.memory.add("checkpoint", f"advanced to: {deps.memory.current_checkpoint()}")
                return "classify"
            if result.status == "OK" and _current_checkpoint_is_last(deps.memory):
                return "finish"
            return "classify"
        if result.needs_fix:
            return "recover"
        return "classify"

    async def finalize_success(state: AgentState) -> AgentState:
        page_state = _parse_page_state(await deps.browser.inspect_state())
        cart: dict[str, Any] = {}
        try:
            cart = json.loads(await deps.browser.cart_snapshot())
        except Exception:
            cart = {}

        if deps.task_kind == "delivery":
            items = cart.get("items", []) if isinstance(cart.get("items", []), list) else []
            exact_ok, exact_reason = _cart_exact_match(deps.profile, cart)
            if not exact_ok and not items:
                blob = f"{cart.get('cart_section_text', '')}\n{page_state.get('body_text', '')}"
                exp = _expected_delivery_items(deps.profile)
                if exp and all(text_matches_target(blob, name) for name, _ in exp):
                    exact_ok, exact_reason = True, "visible text fallback (cart parse empty)"
                    deps.log_event("finalize_soft_match", reason=exact_reason, cart=cart)
            if not exact_ok:
                deps.log_event("finalize_rejected", reason=exact_reason, cart=cart, page_state=page_state)
                return {"stop_reason": f"Final verification failed: {exact_reason}."}
            # cart_snapshot is text-heuristic and may contain menu/category noise.
            # For the final message, only report items that match the user's requested entities.
            expected = _expected_delivery_items(deps.profile)
            item_list: list[str] = []
            if expected:
                for expected_name, expected_qty in expected:
                    matched_qty = 0
                    for it in items:
                        name = _normalize_item_label(it.get("name", ""))
                        if not name:
                            continue
                        if not text_matches_target(name, expected_name):
                            continue
                        qty = it.get("qty", 1)
                        try:
                            qty_int = int(qty)
                        except Exception:
                            qty_int = 1
                        matched_qty += max(1, qty_int)
                    # Prefer showing the user's requested label, not the noisy snapshot label.
                    item_list.append(f"{expected_name} ×{max(expected_qty, matched_qty or expected_qty)}")
            else:
                for it in items[:8]:
                    name = str(it.get("name", "")).strip()
                    qty = it.get("qty", 1)
                    try:
                        qty_int = int(qty)
                    except Exception:
                        qty_int = 1
                    if name:
                        item_list.append(f"{name} ×{max(1, qty_int)}")
            address = str(cart.get("address", "")).strip()
            if item_list:
                finished_message = "В корзине: " + ", ".join(item_list) + "."
            else:
                finished_message = _build_final_message(deps, page_state)
            if address:
                finished_message += f" Адрес: {address}."
            finished_message += " До оплаты я не переходил."
        else:
            finished_message = _build_final_message(deps, page_state)
        deps.log_event("finalized_success", message=finished_message, page_state=page_state)
        return {"completed_successfully": True, "finished_message": finished_message}

    def route_after_functions(state: AgentState) -> str:
        if state.get("completed_successfully") or state.get("stop_reason"):
            return "finish"
        return "classify"

    graph.add_node("bootstrap", bootstrap)
    graph.add_node("plan", plan)
    graph.add_node("ensure_start_page", ensure_start_page)
    graph.add_node("start_executor", start_executor)
    graph.add_node("classify", classify)
    graph.add_node("handle_computer", handle_computer)
    graph.add_node("evaluate", evaluate)
    graph.add_node("recover", recover)
    graph.add_node("finalize_success", finalize_success)
    graph.add_node("handle_functions", handle_functions)
    graph.add_node("nudge", nudge)
    graph.add_node("finish", finish)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "plan")
    graph.add_edge("plan", "ensure_start_page")
    graph.add_edge("ensure_start_page", "start_executor")
    graph.add_edge("start_executor", "classify")
    graph.add_conditional_edges(
        "classify",
        route_from_classify,
        {
            "handle_computer": "handle_computer",
            "handle_functions": "handle_functions",
            "nudge": "nudge",
            "finalize_success": "finalize_success",
            "finish": "finish",
        },
    )
    graph.add_conditional_edges("handle_computer", route_after_computer, {"evaluate": "evaluate", "finish": "finish"})
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"recover": "recover", "classify": "classify", "finalize_success": "finalize_success", "finish": "finish"})
    graph.add_edge("recover", "start_executor")
    graph.add_edge("finalize_success", "finish")
    graph.add_conditional_edges("handle_functions", route_after_functions, {"classify": "classify", "finish": "finish"})
    graph.add_edge("nudge", "classify")
    graph.add_edge("finish", END)
    return graph.compile(debug=False, name="browser-agent")


async def _process_function_call(fc: Any, state: AgentState, deps: AgentDeps, done_verified: bool, step: int) -> tuple[bool, str, bool, str]:
    try:
        args = json.loads(fc.arguments) if fc.arguments else {}
    except Exception:
        args = {}
    name = fc.name
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    deps.console.print(f"  [bold]Step {step}/{state['step_limit']}[/bold] [magenta]{name}[/magenta]({args_str})")
    deps.memory.add("tool", f"{name}({args_str})")
    deps.log_event("tool_called", name=name, args=args)
    page_state = _parse_page_state(await deps.browser.inspect_state())
    page_mode = infer_page_mode(
        current_url=page_state.get("url", ""),
        page_text=page_state.get("body_text", ""),
        flags=page_state.get("flags", {}),
    )
    if name == "ask_user":
        question = args.get("question", "")
        if is_authorization_request(question):
            rejection = (
                "REJECTED. Do not ask the user for permission for ordinary task steps that are already explicitly requested. "
                "Continue autonomously unless you truly need missing information, a login, CAPTCHA handling, payment approval, "
                "or another sensitive confirmation."
            )
            deps.memory.add("tool_result", rejection)
            deps.log_event("tool_rejected", name=name, reason=rejection, question=question)
            deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
            return False, "", done_verified, rejection
        deps.console.print(Panel(question, title="Agent asks", border_style="yellow"))
        answer = Prompt.ask("[bold]Your answer[/bold]")
        deps.memory.add("user", answer)
        deps.log_event("user_prompt", question=question, answer=answer)
        return False, "", done_verified, f"User replied: {answer}"
    if name == "done":
        msg = args.get("message", "(task complete)")
        if deps.memory.has_uncommitted_search():
            rejection = (
                "done() rejected. A search-like field has a typed query that was not committed yet. "
                "Use submit_observed_search on that field, click a visible search/go/suggestion control, "
                "or wait until the results page is visibly loaded before finishing."
            )
            deps.memory.add("tool_result", rejection)
            deps.log_event("done_rejected", message=msg, reason="uncommitted_search")
            return False, "", False, rejection
        if done_verified and _done_gate_satisfied(deps.memory):
            page_state = _parse_page_state(await deps.browser.inspect_state())
            completion_ok, reason = verify_task_completion(
                task=deps.memory.task,
                task_kind=deps.task_kind,
                current_url=page_state.get("url", ""),
                page_text=page_state.get("body_text", ""),
                current_checkpoint=deps.memory.current_checkpoint(),
                result=parse_evaluation(deps.memory.last_evaluation),
                has_uncommitted_search=deps.memory.has_uncommitted_search(),
            )
            if not completion_ok:
                rejection = f"done() rejected. Deterministic verification failed: {reason}."
                deps.memory.add("tool_result", rejection)
                deps.log_event("done_rejected", message=msg, reason=reason, page_state=page_state)
                return False, "", False, rejection
            deps.log_event("done_confirmed", message=msg)
            return True, msg, done_verified, "Confirmed."
        if done_verified and not _done_gate_satisfied(deps.memory):
            rejection = (
                "done() rejected. The task is not yet proven complete. "
                "Do not finish while any requested item, recipient, destination, or stopping condition is still unverified. "
                "Continue working and wait for an evaluator result that marks the final checkpoint COMPLETE and ready_to_finish."
            )
            deps.memory.add("tool_result", rejection)
            deps.log_event("done_rejected", message=msg, reason="gate_not_satisfied")
            return False, "", False, rejection
        deps.memory.add("candidate_done", msg)
        deps.log_event("done_candidate", message=msg)
        return False, "", True, (
            f"Verify that the task is truly complete: \"{msg}\"\n"
            "Take a fresh screenshot and check for visible proof.\n"
            "Only call done() again if the final checkpoint is visibly complete and ready to finish.\n"
            "If proof is missing, continue working."
        )
    if name == "search_web":
        current_domain = _normalize_domain(await deps.browser.current_url())
        if not deps.allow_external_search and current_domain and current_domain not in {"google.com", "bing.com", "yandex.ru"}:
            rejection = "REJECTED. You are already inside a website or web app. Do not switch to external web search. Use the current site's visible UI."
            deps.memory.add("tool_result", rejection)
            deps.log_event("tool_rejected", name=name, reason=rejection)
            deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
            return False, "", done_verified, rejection
    if name in {"navigate", "go_back", "search_web"} and deps.memory.has_uncommitted_search():
        rejection = (
            "REJECTED. Commit or dismiss the active search query first: submit_observed_search, Enter on the search field, "
            "click a visible search/go button, or pick a suggestion. Do not navigate away with an uncommitted search box."
        )
        deps.memory.add("tool_result", rejection)
        deps.log_event("tool_rejected", name=name, reason="uncommitted_search")
        deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
        return False, "", done_verified, rejection
    if name == "navigate":
        current_domain = _normalize_domain(await deps.browser.current_url())
        target_domain = _normalize_domain(args.get("url", ""))
        parsed = urlparse(args.get("url", ""))
        if current_domain and target_domain != current_domain and ((parsed.path not in {"", "/"}) or bool(parsed.query)):
            rejection = "REJECTED. Do not guess deep internal URLs. Open the site's main page and use visible navigation."
            deps.memory.add("tool_result", rejection)
            deps.log_event("tool_rejected", name=name, reason=rejection)
            deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
            return False, "", done_verified, rejection
    if name in {"click_observed", "type_into_observed"}:
        element_id = str(args.get("element_id", ""))
        observed = deps.browser.get_observed_element(element_id) or {}
        label = str(observed.get("label", "")).lower()
        if name == "type_into_observed" and deps.memory.has_uncommitted_search() and not str(args.get("text", "")).strip():
            rejection = (
                "REJECTED. Do not clear an active search field before committing the query. "
                "Use submit_observed_search, click search, or pick a result first."
            )
            deps.memory.add("tool_result", rejection)
            deps.log_event("tool_rejected", name=name, reason="premature_search_clear", element_id=element_id)
            deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
            return False, "", done_verified, rejection
        current_checkpoint = deps.memory.current_checkpoint().lower()
        role = str(observed.get("role", "")).lower()
        search_scope = infer_search_scope(label=label, current_url=page_state.get("url", ""), page_mode=page_mode)
        address_like = is_address_like_field(label, role) or any(token in label for token in ("улиц", "адрес", "дом", "кварт", "подъезд", "достав"))
        checkpoint_is_address = any(token in current_checkpoint for token in ("адрес", "достав"))
        if address_like and not checkpoint_is_address and not task_has_explicit_address(deps.memory.task):
            rejection = (
                "REJECTED. Do not open or fill an address flow yet: the user did not provide an address, "
                "and the current checkpoint is not about address confirmation. Continue inside the current visible workflow "
                "or ask the user only if the task truly cannot proceed without address details."
            )
            deps.memory.add("tool_result", rejection)
            deps.log_event("tool_rejected", name=name, reason=rejection, element_id=element_id, label=label)
            deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
            return False, "", done_verified, rejection
        if deps.profile.target_container and page_mode not in {"cart", "captcha"} and name == "click_observed":
            if is_probable_restaurant_card(label) and not text_matches_target(label, deps.profile.target_container):
                visible_target = text_matches_target(page_state.get("body_text", ""), deps.profile.target_container)
                if visible_target:
                    rejection = (
                        f"REJECTED. The target container '{deps.profile.target_container}' is already visible. "
                        "Do not open a different card."
                    )
                    deps.memory.add("tool_result", rejection)
                    deps.log_event("tool_rejected", name=name, reason=rejection, element_id=element_id, label=label)
                    deps.console.print(f"  [yellow]-> {rejection[:120]}[/yellow]")
                    return False, "", done_verified, rejection
    click_observed_meta: dict[str, str] | None = None
    type_observed_meta: dict[str, str] | None = None
    if name == "click_observed":
        element_id = str(args.get("element_id", ""))
        obs_pre = deps.browser.get_observed_element(element_id) or {}
        click_observed_meta = {
            "label": str(obs_pre.get("label", "")),
            "role": str(obs_pre.get("role", "")),
        }
    elif name == "type_into_observed":
        element_id = str(args.get("element_id", ""))
        obs_pre = deps.browser.get_observed_element(element_id) or {}
        type_observed_meta = {
            "label": str(obs_pre.get("label", "")),
            "role": str(obs_pre.get("role", "")),
        }

    fn = getattr(deps.browser, name, None)
    try:
        result = await fn(**args) if fn else f"Unknown tool: {name}"
    except Exception as exc:
        result = f"Tool {name} failed: {exc}"

    if name == "type_into_observed":
        if type_observed_meta is not None:
            lbl = type_observed_meta["label"]
            rrole = type_observed_meta["role"]
        else:
            obs = deps.browser.get_observed_element(str(args.get("element_id", ""))) or {}
            lbl = str(obs.get("label", ""))
            rrole = str(obs.get("role", ""))
        typed = str(args.get("text", ""))
        if is_search_like_field(lbl, rrole) and typed.strip():
            ps = _parse_page_state(await deps.browser.inspect_state())
            deps.memory.start_or_update_search_commit(
                element_id=str(args.get("element_id", "")),
                query=typed,
                url=str(ps.get("url", "")),
                body_text=str(ps.get("body_text", "")),
            )
        elif is_address_like_field(lbl, rrole) and typed.strip():
            deps.memory.start_pending_address(typed)
    elif name == "submit_observed_search":
        deps.memory.end_search_commit()
    elif name == "click_observed" and click_observed_meta is not None:
        ps = _parse_page_state(await deps.browser.inspect_state())
        lbl = click_observed_meta["label"]
        rrole = click_observed_meta["role"]
        if deps.memory.has_uncommitted_search():
            if is_search_commit_control_click(lbl, rrole) or is_search_suggestion_pick(rrole):
                deps.memory.end_search_commit()
            elif str(ps.get("url", "")).strip() != (deps.memory.pending_search_url_snapshot or "").strip():
                deps.memory.end_search_commit()
        if deps.memory.has_pending_address() and is_address_commit_control(lbl, rrole):
            deps.memory.end_pending_address()

    deps.memory.add("tool_result", result)
    deps.log_event("tool_result", name=name, result=result)
    deps.console.print(f"  [green]-> {result[:120]}[/green]")
    return False, "", done_verified, f"{result}\nPage loaded. Take a screenshot to see the current state."


async def run_agent_graph(task: str) -> None:
    console = Console()
    profile = build_task_profile(task)
    task_domains = profile.domains | _extract_domains(task)
    task_kind = profile.kind
    deps = AgentDeps(
        console=console,
        browser=BrowserSession(BrowserConfig(
            headless=os.getenv("HEADLESS", "0").strip() == "1",
            slow_mo_ms=int(os.getenv("SLOW_MO_MS", "100")),
            # Default to Playwright's bundled Chromium (more reliable than real Chrome channel).
            use_chrome=os.getenv("USE_CHROME", "1").strip() == "1",
            chrome_profile=os.getenv("CHROME_PROFILE", "").strip(),
            chrome_profile_directory=os.getenv("CHROME_PROFILE_DIRECTORY", "").strip(),
            persist_session=os.getenv("PERSIST_SESSION", "1").strip() == "1",
            persist_profile_dir=os.getenv("PERSIST_PROFILE_DIR", ".agent_profile").strip() or ".agent_profile",
            launch_timeout_s=float(os.getenv("BROWSER_LAUNCH_TIMEOUT_S", "20").strip() or "20"),
        )),
        llm=LLMClient(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini")),
        planner=TaskPlanner(LLMClient(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))),
        checkpoint_planner=CheckpointPlanner(LLMClient(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))),
        summarizer=HistorySummarizer(LLMClient(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))),
        evaluator=TaskEvaluator(LLMClient(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))),
        memory=TaskMemory(task=task),
        soft_max=int(os.getenv("MAX_STEPS", "60")),
        hard_max=int(os.getenv("HARD_MAX_STEPS", "120")),
        extend_by=int(os.getenv("STEP_EXTENSION", "20")),
        compact_every=int(os.getenv("COMPACT_EVERY_STEPS", "16")),
        keep_browser_open=os.getenv("KEEP_BROWSER_OPEN", "1").strip() == "1",
        startup_timeout=int(os.getenv("STARTUP_HELPER_TIMEOUT", "30")),
        allow_external_search=os.getenv("ALLOW_EXTERNAL_SEARCH", "0").strip() == "1",
        trace_path=Path(os.getenv("TRACE_DIR", ".agent_runs")) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl",
        task_domains=task_domains,
        task_kind=task_kind,
        requested_entities=profile.requested_entities,
        profile=profile,
    )
    deps.trace_path.parent.mkdir(parents=True, exist_ok=True)
    graph = await _build_graph(deps)
    state = await graph.ainvoke({"task": task})
    try:
        if state.get("completed_successfully"):
            console.print(Panel(state.get("finished_message", "(done)"), title="Done", border_style="green"))
        elif state.get("stop_reason"):
            console.print(Panel(state["stop_reason"], title="Stopped", border_style="red"))
    except Exception:
        pass
    finally:
        try:
            if state.get("completed_successfully") and deps.keep_browser_open:
                console.print("[dim]Browser session is complete. Press Enter when you want to close the agent and release the browser.[/dim]")
                try:
                    Prompt.ask("[bold]Press Enter to close the agent session[/bold]", default="")
                except EOFError:
                    pass
            await deps.browser.close()
        except Exception:
            pass
