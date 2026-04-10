"""Planner, summarizer, evaluator, and checkpoint roles for the browser agent."""

from __future__ import annotations

import json
import re

from agent.policy import EvaluationResult, parse_evaluation
from agent.prompts import (
    get_checkpoint_prompt,
    get_evaluator_prompt,
    get_planner_prompt,
    get_summarizer_prompt,
)


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


class TaskPlanner:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    async def plan(self, task_context: str) -> str:
        return await self.llm.complete_text(
            system_prompt=get_planner_prompt(),
            user_input=task_context,
        )


class CheckpointPlanner:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    async def build(self, task: str) -> list[str]:
        text = await self.llm.complete_text(
            system_prompt=get_checkpoint_prompt(),
            user_input=task,
        )
        try:
            data = json.loads(_extract_json_block(text))
            checkpoints = [str(item).strip() for item in data.get("checkpoints", []) if str(item).strip()]
            return checkpoints[:6]
        except Exception:
            checkpoints = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    checkpoints.append(stripped[2:].strip())
            return checkpoints[:6]


class HistorySummarizer:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    async def summarize(self, task_context: str) -> str:
        return await self.llm.complete_text(
            system_prompt=get_summarizer_prompt(),
            user_input=task_context,
        )


class TaskEvaluator:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    async def evaluate(
        self,
        task_context: str,
        current_checkpoint: str,
        recent_actions: str,
        screenshot_url: str,
    ) -> EvaluationResult:
        payload = (
            f"{task_context}\n\n"
            f"Current checkpoint:\n{current_checkpoint}\n\n"
            f"Recent actions:\n{recent_actions}\n\n"
            "Inspect the screenshot and determine whether the last actions improved the exact task outcome."
        )
        text = await self.llm.complete_text_with_image(
            system_prompt=get_evaluator_prompt(),
            user_input=payload,
            screenshot_url=screenshot_url,
        )
        return parse_evaluation(text)
