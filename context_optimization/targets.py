from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from .io import content_hash
from .schema import DecisionResult, EvaluationUnit


class DecisionTarget(Protocol):
    target_id: str

    def evaluate(self, unit: EvaluationUnit, source: str | None = None) -> DecisionResult: ...

    def evaluate_many(self, units: list[EvaluationUnit]) -> list[DecisionResult]: ...


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    for attempt in range(3):
        try:
            response = client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
            continue
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
            response.raise_for_status()
            return response
        time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


class JevTarget:
    target_id = "jev"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        api_key = os.environ[api_key_env]
        self.base_url = base_url or os.getenv(
            "TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"
        )
        self.model = model or os.getenv("TYPESAFE_MODEL", "jev-1.13.0")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    @staticmethod
    def _question_payload(unit: EvaluationUnit) -> dict[str, Any]:
        if unit.primitive == "choice":
            return {
                "type": "choice",
                "instructions": unit.instructions,
                "criteria": unit.branches,
            }
        return {
            "type": "noul",
            "instructions": unit.instructions,
            "criteria": {"true": unit.branches["true"], "false": unit.branches["false"]},
        }

    def build_payload_many(
        self, units: list[EvaluationUnit], source: str | None = None
    ) -> dict[str, Any]:
        if not units:
            raise ValueError("units cannot be empty")
        if any(unit.source != units[0].source for unit in units):
            raise ValueError("batched units must share one source")
        return {
            "state": source if source is not None else units[0].source,
            "model": self.model,
            "questions": {
                f"decision_{index}": self._question_payload(unit)
                for index, unit in enumerate(units)
            },
        }

    def evaluate(self, unit: EvaluationUnit, source: str | None = None) -> DecisionResult:
        payload = self.build_payload_many([unit], source=source)
        return self._evaluate_payload([unit], payload)[0]

    def evaluate_many(self, units: list[EvaluationUnit]) -> list[DecisionResult]:
        return self._evaluate_payload(units, self.build_payload_many(units))

    def _evaluate_payload(
        self, units: list[EvaluationUnit], payload: dict[str, Any]
    ) -> list[DecisionResult]:
        start = time.perf_counter()
        response = _request_with_retry(self.client, "POST", self.base_url, json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        body = response.json()
        results = []
        for index, unit in enumerate(units):
            answer = body["answers"][f"decision_{index}"]
            if unit.primitive == "choice":
                raw = {str(key): float(value) for key, value in answer["probabilities"].items()}
                total = sum(raw.values())
                probabilities = {key: value / total for key, value in raw.items()}
                native_selected = str(answer["choice"])
                selected = max(probabilities, key=probabilities.get)
                confidence = float(answer["confidence"])
            else:
                probability_true = float(answer["noul"])
                probabilities = {"false": 1.0 - probability_true, "true": probability_true}
                native_selected = None
                selected = max(probabilities, key=probabilities.get)
                confidence = None
            results.append(
                DecisionResult(
                    unit_id=unit.unit_id,
                    target_id=self.target_id,
                    selected=selected,
                    probabilities=probabilities,
                    native_confidence=confidence,
                    latency_ms=latency_ms,
                    input_tokens=(body.get("usage", {}).get("input_tokens") if index == 0 else 0),
                    model_revision=str(body.get("model", self.model)),
                    request_hash=content_hash(
                        {"target": self.target_id, "payload": payload, "unit_id": unit.unit_id}
                    ),
                    metadata={"native_selected": native_selected, "batch_size": len(units)},
                )
            )
        return results


class HttpDecisionTarget:
    def __init__(self, target_id: str, base_url: str, *, timeout: float = 120.0) -> None:
        self.target_id = target_id
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def evaluate(self, unit: EvaluationUnit, source: str | None = None) -> DecisionResult:
        payload = {
            "unit": unit.model_dump(mode="json"),
            "source": source if source is not None else unit.source,
        }
        start = time.perf_counter()
        response = _request_with_retry(
            self.client, "POST", f"{self.base_url}/evaluate", json=payload
        )
        latency_ms = (time.perf_counter() - start) * 1000
        body = response.json()
        body.update(
            {
                "unit_id": unit.unit_id,
                "target_id": self.target_id,
                "latency_ms": latency_ms,
                "request_hash": content_hash({"target": self.target_id, "payload": payload}),
            }
        )
        return DecisionResult.model_validate(body)

    def evaluate_many(self, units: list[EvaluationUnit]) -> list[DecisionResult]:
        return [self.evaluate(unit) for unit in units]


class CallableTarget:
    def __init__(
        self,
        target_id: str,
        callback: Callable[[EvaluationUnit, str], DecisionResult],
    ) -> None:
        self.target_id = target_id
        self.callback = callback

    def evaluate(self, unit: EvaluationUnit, source: str | None = None) -> DecisionResult:
        return self.callback(unit, source if source is not None else unit.source)

    def evaluate_many(self, units: list[EvaluationUnit]) -> list[DecisionResult]:
        return [self.evaluate(unit) for unit in units]


class CachedTarget:
    def __init__(self, target: DecisionTarget, cache_dir: str | Path) -> None:
        self.target = target
        self.target_id = target.target_id
        self.cache_dir = Path(cache_dir) / "targets" / self.target_id

    def _path(self, unit: EvaluationUnit, source: str) -> Path:
        key = content_hash(
            {
                "target": self.target_id,
                "unit": unit,
                "source": source,
            }
        )
        return self.cache_dir / f"{key}.json"

    def evaluate(self, unit: EvaluationUnit, source: str | None = None) -> DecisionResult:
        effective_source = source if source is not None else unit.source
        path = self._path(unit, effective_source)
        if path.exists():
            return DecisionResult.model_validate_json(path.read_text(encoding="utf-8"))
        result = self.target.evaluate(unit, effective_source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json() + "\n", encoding="utf-8")
        return result

    def evaluate_many(self, units: list[EvaluationUnit]) -> list[DecisionResult]:
        cached: dict[str, DecisionResult] = {}
        missing = []
        paths = {}
        for unit in units:
            path = self._path(unit, unit.source)
            paths[unit.unit_id] = path
            if path.exists():
                cached[unit.unit_id] = DecisionResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            else:
                missing.append(unit)
        if missing:
            for result in self.target.evaluate_many(missing):
                path = paths[result.unit_id]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.model_dump_json() + "\n", encoding="utf-8")
                cached[result.unit_id] = result
        return [cached[unit.unit_id] for unit in units]
