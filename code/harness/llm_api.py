#!/usr/bin/env python3
"""Small, dependency-free clients for API-backed replication experiments.

The replication code needs to call models exposed through OpenRouter, Microsoft
Foundry's OpenAI-compatible endpoint, and Foundry's Anthropic-compatible
endpoint.  This module keeps those transport differences out of experiment
logic and, importantly, never accepts literal API keys in JSON configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
UNUSABLE_FINISH_REASONS = {"error"}
UNCACHEABLE_FINISH_REASONS = {"error", "length", "max_tokens"}


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _join_url(base_url: str, endpoint_path: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") in {None, "text"}
        )
    return str(content or "")


def _structured_text(value: Any) -> str:
    """Extract human-readable text from provider reasoning structures.

    OpenRouter normally returns reasoning in ``message.reasoning``, but some
    routed providers use content blocks or a ``reasoning_details`` array.  We
    retain the original structures separately and use this helper only for the
    convenient, analysis-ready text field.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_structured_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "thinking", "reasoning", "content", "summary"):
            if value.get(key) not in (None, ""):
                return _structured_text(value[key])
    return ""


def _message_reasoning(message: dict[str, Any]) -> tuple[str, Any]:
    """Return normalized reasoning text and the unmodified provider details."""
    details = message.get("reasoning_details")
    for key in ("reasoning", "reasoning_content", "thinking", "analysis"):
        text = _structured_text(message.get(key))
        if text:
            return text, details
    return _structured_text(details), details


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved non-secret provider settings.

    ``protocol`` is either ``openai_chat`` or ``anthropic_messages``.
    ``auth_style`` controls only the HTTP header used for the key.
    """

    name: str
    protocol: str
    base_url: str
    api_key_env: str
    auth_style: str = "bearer"
    endpoint_path: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 120
    max_retries: int = 4

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "ProviderConfig":
        protocol = str(raw.get("protocol", "openai_chat"))
        if protocol not in {"openai_chat", "anthropic_messages"}:
            raise ValueError(f"Unsupported protocol for provider {name}: {protocol}")

        base_url = str(raw.get("base_url", "")).strip()
        base_url_env = str(raw.get("base_url_env", "")).strip()
        if base_url_env:
            base_url = os.environ.get(base_url_env, base_url).strip()
        if not base_url:
            raise RuntimeError(
                f"Provider {name} needs base_url or a populated base_url_env."
            )

        default_path = (
            "chat/completions" if protocol == "openai_chat" else "v1/messages"
        )
        return cls(
            name=name,
            protocol=protocol,
            base_url=base_url,
            api_key_env=str(raw.get("api_key_env", "EXPERIMENT_LLM_API_KEY")),
            auth_style=str(raw.get("auth_style", "bearer")),
            endpoint_path=str(raw.get("endpoint_path", default_path)),
            extra_headers={
                str(key): str(value)
                for key, value in raw.get("extra_headers", {}).items()
            },
            timeout_seconds=int(raw.get("timeout_seconds", 120)),
            max_retries=int(raw.get("max_retries", 4)),
        )

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Provider {self.name} requires environment variable "
                f"{self.api_key_env}."
            )
        return key

    def redacted(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_set": bool(os.environ.get(self.api_key_env)),
            "auth_style": self.auth_style,
            "endpoint_path": self.endpoint_path,
        }

    def cache_identity(self) -> dict[str, Any]:
        """Return only stable, non-secret fields that affect the response."""
        return {
            "name": self.name,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "auth_style": self.auth_style,
            "endpoint_path": self.endpoint_path,
            "extra_headers": self.extra_headers,
        }


@dataclass
class ChatApiClient:
    provider: ProviderConfig
    cache_dir: Path

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.provider.extra_headers}
        auth_style = self.provider.auth_style.lower()
        if auth_style == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_style == "api-key":
            headers["api-key"] = api_key
        elif auth_style == "x-api-key":
            headers["x-api-key"] = api_key
        else:
            raise ValueError(
                f"Unsupported auth_style for {self.provider.name}: "
                f"{self.provider.auth_style}"
            )
        return headers

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        max_tokens_field: str = "max_tokens",
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.provider.protocol == "anthropic_messages":
            system_parts = [
                message["content"]
                for message in messages
                if message.get("role") == "system"
            ]
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    message for message in messages if message.get("role") != "system"
                ],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)
        else:
            payload = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens_field not in {"", "none", "omit"}:
                payload[max_tokens_field] = max_tokens
            if seed is not None:
                payload["seed"] = seed
        if extra_body:
            payload.update(extra_body)
        return payload

    def _parse_response(self, data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.provider.protocol == "anthropic_messages":
            content = data.get("content", [])
            output = _content_to_text(content)
            finish_reason = data.get("stop_reason")
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            reasoning_details = [
                block
                for block in content
                if isinstance(block, dict)
                and block.get("type") in {"thinking", "redacted_thinking"}
            ]
            reasoning = _structured_text(reasoning_details)
        else:
            choice = data["choices"][0]
            assistant_message = choice["message"]
            output = _content_to_text(assistant_message.get("content"))
            finish_reason = choice.get("finish_reason")
            reasoning, reasoning_details = _message_reasoning(assistant_message)
        return output, {
            "provider": self.provider.name,
            "usage": data.get("usage", {}),
            "response_id": data.get("id"),
            "response_model": data.get("model"),
            "finish_reason": finish_reason,
            # These fields are deliberately kept in the prediction audit log.
            # ``planned_items.jsonl`` stores the exact prompt; retaining the
            # assistant message and separate reasoning payload here makes each
            # response independently auditable without reconstructing an API
            # call or relying on provider-side retention.
            "assistant_message": assistant_message,
            "reasoning": reasoning,
            "reasoning_details": reasoning_details,
        }

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = 0.0,
        max_tokens: int = 512,
        max_tokens_field: str = "max_tokens",
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = self.build_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tokens_field=max_tokens_field,
            seed=seed,
            extra_body=extra_body,
        )
        cache_key = stable_hash(
            {
                "provider": self.provider.cache_identity(),
                "payload": payload,
            }
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            cached_meta = cached.get("meta", {})
            if cached_meta.get("finish_reason") not in UNCACHEABLE_FINISH_REASONS:
                return cached["output"], {**cached_meta, "cache_hit": True}
            # Error and length-limited responses must never become permanent
            # cache hits. A later request can use a repaired provider response
            # or a larger output ceiling.
            cache_path.unlink()

        api_key = self.provider.api_key()
        request = urllib.request.Request(
            _join_url(self.provider.base_url, self.provider.endpoint_path),
            data=json.dumps(payload).encode(),
            headers=self._headers(api_key),
            method="POST",
        )
        last_error = "API request failed"
        for attempt in range(self.provider.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.provider.timeout_seconds
                ) as response:
                    data = json.loads(response.read().decode())
                output, meta = self._parse_response(data)
                meta = {**meta, "model": model, "cache_hit": False}
                if meta.get("finish_reason") in UNUSABLE_FINISH_REASONS:
                    raise RuntimeError(
                        "Provider returned an unusable completion: "
                        f"finish_reason={meta.get('finish_reason')}"
                    )
                if meta.get("finish_reason") not in UNCACHEABLE_FINISH_REASONS:
                    write_json(cache_path, {"output": output, "meta": meta})
                return output, meta
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:1000]
                last_error = f"HTTP {exc.code}: {body}".replace(api_key, "<redacted>")
                if exc.code not in RETRYABLE_STATUS_CODES:
                    break
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    delay = 2**attempt
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc).replace(api_key, "<redacted>")
                delay = 2**attempt
            if attempt + 1 < self.provider.max_retries:
                time.sleep(min(delay, 60))
        raise RuntimeError(last_error)


def build_provider_clients(
    providers: dict[str, dict[str, Any]], cache_root: Path
) -> dict[str, ChatApiClient]:
    if not providers:
        raise RuntimeError("Configuration must define at least one provider.")
    clients = {}
    for name, raw in providers.items():
        provider = ProviderConfig.from_dict(name, raw)
        clients[name] = ChatApiClient(
            provider=provider,
            cache_dir=cache_root / name,
        )
    return clients
