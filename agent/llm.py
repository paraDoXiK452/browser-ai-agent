"""LLM client for browser-agent runtime."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from dotenv import load_dotenv

from agent.tools import CUSTOM_TOOLS

MAX_RETRIES = 4


async def _call_with_retry(coro_factory, retries: int = MAX_RETRIES) -> Any:
    """Call an async factory, retrying on rate-limit errors."""
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = 3.0 * (attempt + 1)
                match = re.search(r"try again in ([\d.]+)s", str(exc))
                if match:
                    wait = float(match.group(1)) + 0.5
                if attempt < retries:
                    print(f"  [Rate limit] waiting {wait:.1f}s (attempt {attempt + 1}/{retries})...")
                    await asyncio.sleep(wait)
                    continue
            raise


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI

        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._tools = [{"type": "computer"}] + CUSTOM_TOOLS

    async def first_call(self, system_prompt: str, task: str, screenshot_url: str | None = None) -> Any:
        input_items: list[Any] = []
        if screenshot_url:
            input_items.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task},
                    {"type": "input_image", "image_url": screenshot_url, "detail": "original"},
                ],
            })
        else:
            input_items = task  # type: ignore[assignment]

        return await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            tools=self._tools,
            input=input_items,
        ))

    async def restart_from_context(self, system_prompt: str, task_context: str, screenshot_url: str) -> Any:
        return await self.first_call(
            system_prompt=system_prompt,
            task=task_context,
            screenshot_url=screenshot_url,
        )

    async def send_function_outputs(self, prev_response_id: str, outputs: list[dict]) -> Any:
        input_items = [
            {"type": "function_call_output", "call_id": o["call_id"], "output": o["output"]}
            for o in outputs
        ]
        return await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            previous_response_id=prev_response_id,
            tools=self._tools,
            input=input_items,
        ))

    async def send_tool_outputs(self, prev_response_id: str, outputs: list[dict[str, Any]]) -> Any:
        return await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            previous_response_id=prev_response_id,
            tools=self._tools,
            input=outputs,
        ))

    async def nudge(self, prev_response_id: str, severity: int = 1) -> Any:
        if severity >= 2:
            msg = (
                "You have stalled multiple times. "
                "Recover generically: take a fresh screenshot, inspect the visible page, "
                "then do one concrete next action. Prefer one of: scroll, open a visible menu, "
                "focus a visible search or input field, go back, or navigate only if the current page is clearly wrong."
            )
        else:
            msg = (
                "Stop describing plans. Take a fresh screenshot, inspect the page, and continue with concrete browser actions. "
                "Prefer a short safe batch of 2 to 4 related actions when the UI is obvious, but after typing into a field or changing page state, verify before continuing."
            )
        return await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            previous_response_id=prev_response_id,
            tools=self._tools,
            input=msg,
        ))

    async def send_text_input(self, prev_response_id: str, text: str) -> Any:
        return await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            previous_response_id=prev_response_id,
            tools=self._tools,
            input=text,
        ))

    async def complete_text(self, system_prompt: str, user_input: str) -> str:
        response = await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input,
        ))
        return self._extract_text(response)

    async def complete_text_with_image(self, system_prompt: str, user_input: str, screenshot_url: str) -> str:
        response = await _call_with_retry(lambda: self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_input},
                    {"type": "input_image", "image_url": screenshot_url, "detail": "original"},
                ],
            }],
        ))
        return self._extract_text(response)

    def _extract_text(self, response: Any) -> str:
        chunks: list[str] = []
        for item in response.output:
            if item.type != "message":
                continue
            for block in item.content:
                text = getattr(block, "text", "")
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
