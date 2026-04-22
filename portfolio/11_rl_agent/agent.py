"""Minimal ReAct-style tool-use agent loop and deterministic eval harness.

Deliberately torch-free: the agent's *policy* is an abstract callable so we
can unit-test the loop with a hand-written policy, then swap in an LLM at
demo time.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Tool = Callable[[str], str]

# -----------------------------------------------------------------------------
# Tools


def calculator(expr: str) -> str:
    """Evaluate a simple arithmetic expression. Supports + - * / ** and parentheses."""
    expr = expr.strip()
    if not re.fullmatch(r"[0-9.+\-*/() ]+|\*\*", expr.replace("**", "")):
        return f"error: invalid characters in {expr!r}"
    try:
        # Use a restricted eval via AST for safety; small arithmetic only.
        import ast

        tree = ast.parse(expr, mode="eval")
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _ev(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_ev(node.left), _ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
                return allowed_ops[type(node.op)](_ev(node.operand))
            raise ValueError(f"unsupported node: {type(node).__name__}")

        return str(_ev(tree))
    except Exception as exc:  # pragma: no cover - intentional catch-all
        return f"error: {exc}"


def build_retriever(corpus: dict[str, str]) -> Tool:
    """Return a tool that does keyword-lookup over `corpus` (dict name → text).

    Scores each entry by count of query tokens that appear in the body. Ties
    break alphabetically so the harness is deterministic.
    """

    def _retrieve(query: str) -> str:
        q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if t]
        if not q_tokens:
            return "no results"
        scored: list[tuple[int, str, str]] = []
        for name, body in corpus.items():
            body_lower = body.lower()
            score = sum(t in body_lower for t in q_tokens)
            if score > 0:
                scored.append((score, name, body))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            return "no results"
        score, name, body = scored[0]
        return f"{name}: {body}"

    return _retrieve


# -----------------------------------------------------------------------------
# Agent loop


@dataclass
class AgentStep:
    thought: str
    action_name: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    success: bool


Policy = Callable[[str, list[AgentStep]], tuple[str, str, str]]
# policy(prompt, history) -> (thought, action_name, action_input)
# action_name may be "final" in which case action_input is the final answer.


@dataclass
class AgentConfig:
    max_steps: int = 6
    tools: dict[str, Tool] = field(default_factory=dict)


def run_agent(prompt: str, policy: Policy, config: AgentConfig) -> AgentResult:
    """Run a ReAct-style loop until the policy emits `final` or we hit max_steps."""
    history: list[AgentStep] = []
    for _ in range(config.max_steps):
        thought, action_name, action_input = policy(prompt, history)
        if action_name == "final":
            history.append(AgentStep(thought, action_name, action_input, action_input))
            return AgentResult(answer=action_input, steps=history, success=True)
        tool = config.tools.get(action_name)
        observation = (
            tool(action_input) if tool is not None else f"error: unknown tool {action_name!r}"
        )
        history.append(AgentStep(thought, action_name, action_input, observation))
    return AgentResult(answer="", steps=history, success=False)


# -----------------------------------------------------------------------------
# Eval harness


@dataclass
class AgentTask:
    prompt: str
    reference: str
    tolerance: float = 0.0  # for numeric answers


@dataclass
class AgentEvalReport:
    success_rate: float
    n: int
    per_task: list[tuple[AgentTask, AgentResult, bool]]


def _grade(task: AgentTask, result: AgentResult) -> bool:
    if not result.success:
        return False
    predicted = result.answer.strip()
    if task.tolerance > 0:
        try:
            return math.isclose(float(predicted), float(task.reference), abs_tol=task.tolerance)
        except ValueError:
            return False
    return predicted == task.reference.strip()


def evaluate_agent(
    tasks: Sequence[AgentTask], policy: Policy, config: AgentConfig
) -> AgentEvalReport:
    per_task: list[tuple[AgentTask, AgentResult, bool]] = []
    correct = 0
    for task in tasks:
        result = run_agent(task.prompt, policy, config)
        ok = _grade(task, result)
        correct += int(ok)
        per_task.append((task, result, ok))
    return AgentEvalReport(
        success_rate=correct / max(len(tasks), 1), n=len(tasks), per_task=per_task
    )


# -----------------------------------------------------------------------------
# Example hand-written policies (mostly for tests)


def make_math_and_lookup_policy(corpus_tool_name: str = "retrieve") -> Policy:
    """A deterministic policy that routes numeric-looking questions to the calculator
    and text-looking questions to retrieval. Gives the harness a working unit-test
    target without requiring an LLM.
    """

    def policy(prompt: str, history: list[AgentStep]) -> tuple[str, str, str]:
        # After the first observation we always finalise.
        if history:
            return ("use observation", "final", history[-1].observation.split(":", 1)[-1].strip())
        is_math = bool(re.fullmatch(r"[0-9.+\-*/()\s]+(=?\?)?", prompt.strip()))
        if is_math:
            expr = prompt.rstrip("?=").strip()
            return ("arithmetic", "calculator", expr)
        return ("lookup", corpus_tool_name, prompt)

    return policy


def _extract_numeric(text: str) -> str | None:
    """Return the first numeric token in `text`, else None."""
    match = re.search(r"-?\d+\.?\d*", text)
    return match.group(0) if match else None


def make_numeric_policy() -> Policy:
    """Policy that finalises with the first numeric token from any tool output."""

    def policy(prompt: str, history: list[AgentStep]) -> tuple[str, str, str]:
        if history:
            num = _extract_numeric(history[-1].observation)
            return ("extract numeric", "final", num or "")
        return ("arithmetic", "calculator", prompt)

    return policy


def make_fixed_trace_policy(trace: Sequence[tuple[str, str, str]]) -> Policy:
    """Replay a pre-recorded trace step-by-step — useful for testing the loop shape."""
    i = {"t": 0}

    def policy(_prompt: str, _history: list[AgentStep]) -> tuple[str, str, str]:
        step = trace[i["t"]]
        i["t"] = (i["t"] + 1) % len(trace)
        return step

    return policy


def _make_toolkit(corpus: dict[str, str] | None = None) -> dict[str, Tool]:
    corpus = corpus or {
        "paris": "Paris is the capital of France.",
        "london": "London is the capital of the United Kingdom.",
        "tokyo": "Tokyo is the capital of Japan.",
    }
    return {"calculator": calculator, "retrieve": build_retriever(corpus)}


def default_config(corpus: dict[str, str] | None = None) -> AgentConfig:
    return AgentConfig(max_steps=6, tools=_make_toolkit(corpus))


def _sample_eval_tasks() -> list[AgentTask]:
    return [
        AgentTask("2 + 3 * 4", "14"),
        AgentTask("(10 - 4) / 2", "3.0"),
        AgentTask("What is the capital of France?", "Paris is the capital of France."),
        AgentTask("What is the capital of Japan?", "Tokyo is the capital of Japan."),
    ]


def _extract_answer_from_observation(obs: str) -> str:
    # "name: body" → body; else the whole thing.
    if ":" in obs:
        return obs.split(":", 1)[1].strip()
    return obs.strip()


if __name__ == "__main__":
    tasks = _sample_eval_tasks()
    policy = make_math_and_lookup_policy()
    report = evaluate_agent(tasks, policy, default_config())
    print(f"success rate: {report.success_rate:.2f} over {report.n} tasks")
    for t, r, ok in report.per_task:
        print(f"  [{'OK' if ok else 'FAIL'}] {t.prompt!r} -> {r.answer!r}")


# Export a small set of symbols for external use.
_ = (math, Any)
