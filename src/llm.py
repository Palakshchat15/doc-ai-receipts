"""Minimal Ollama chat client with structured outputs. No mock fallback: if Ollama is down we raise."""
from __future__ import annotations

import time

import requests

from .config import load_config


class OllamaUnavailable(RuntimeError):
    pass


def _base() -> str:
    return load_config()["llm"]["base_url"].rstrip("/")


def list_models() -> list[str]:
    try:
        r = requests.get(f"{_base()}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise OllamaUnavailable(
            f"Ollama is not reachable at {_base()} ({type(e).__name__}). "
            "Start it (open the Ollama app or run `ollama serve`) and try again.") from e
    return [m["name"] for m in r.json().get("models", [])]


def check_ollama(model: str | None = None) -> None:
    """Raise OllamaUnavailable with a clear message if the server or the model is missing."""
    names = set(list_models())
    if model and model not in names and f"{model}:latest" not in names:
        raise OllamaUnavailable(f"Model '{model}' is not pulled. Run: ollama pull {model}")


def chat(messages: list[dict], model: str, fmt: dict | str | None = None,
         num_ctx: int | None = None, num_predict: int | None = None) -> dict:
    """One non-streaming /api/chat call. `fmt` is a JSON schema (structured outputs) or 'json'."""
    cfg = load_config()["llm"]
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": cfg["keep_alive"],
        "options": {
            "temperature": cfg["temperature"],
            "seed": cfg["seed"],
            "num_ctx": num_ctx or cfg["num_ctx"],
            "num_predict": num_predict or cfg["num_predict"],
        },
    }
    if fmt is not None:
        body["format"] = fmt
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{_base()}/api/chat", json=body, timeout=cfg["request_timeout_s"])
    except requests.ConnectionError as e:
        raise OllamaUnavailable(f"Ollama is not reachable at {_base()}. Start Ollama and retry.") from e
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    d = r.json()
    return {
        "content": d["message"]["content"],
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "completion_tokens": d.get("eval_count", 0),
        "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
        "eval_s": d.get("eval_duration", 0) / 1e9,
        "load_s": d.get("load_duration", 0) / 1e9,
        "wall_s": time.perf_counter() - t0,
        "done_reason": d.get("done_reason"),
        "model": model,
    }
