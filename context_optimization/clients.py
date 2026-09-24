from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "local",
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def models(self) -> list[dict[str, Any]]:
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()["data"]

    def structured_chat(
        self,
        *,
        model: str,
        system: str,
        user: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
        }
        if seed is not None:
            payload["seed"] = seed
        last_content = ""
        for attempt, token_budget in enumerate((max_tokens, max_tokens * 2)):
            payload["max_tokens"] = token_budget
            if seed is not None:
                payload["seed"] = seed + attempt
            response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            last_content = body["choices"][0]["message"]["content"]
            try:
                return json.loads(last_content), body.get("usage", {})
            except json.JSONDecodeError:
                continue
        raise ValueError(f"structured response was invalid after retry: {last_content!r}")
