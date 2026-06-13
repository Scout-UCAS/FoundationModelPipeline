from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleEvaluator:
    name: str = "example-evaluator"

    def evaluate(self, prompt: str, response: str) -> dict[str, float]:
        exact = 1.0 if prompt.strip().lower() in response.strip().lower() else 0.0
        return {"contains_prompt": exact, "response_length": float(len(response))}


def build_example_evaluator() -> ExampleEvaluator:
    return ExampleEvaluator()

