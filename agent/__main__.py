"""CLI entry point for the browser AI agent."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent.graph_runtime import run_agent_graph


def main() -> None:
    load_dotenv()
    console = Console()
    console.print(Panel(
        "[bold]Browser AI Agent[/bold]\n"
        "Describe a task in text and watch the agent solve it in a visible browser.\n"
        "Press Ctrl+C to stop.",
        border_style="blue",
    ))
    task = Prompt.ask("\n[bold]Describe the task[/bold]")
    if not task.strip():
        console.print("[red]Empty task.[/red]")
        return
    runner: asyncio.Runner | None = None
    try:
        runner = asyncio.Runner()
        runner.run(run_agent_graph(task.strip()))
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent stopped by user.[/yellow]")
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
