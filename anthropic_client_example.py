"""
Open LLM Gateway - Anthropic(Claude) 請求範例（透過 Gateway 的 OpenAI 相容介面）

本腳本示範如何用 HTTP 直接呼叫 Gateway 的 OpenAI 相容端點，並指定 `claude/...` 模型，
以驗證 main.py 對 Claude（Anthropic）轉發的非串流與串流能力。

前置條件：
1) Gateway(main.py) 已啟動，且對外提供 /v1/chat/completions
2) main.py 所在環境已設定 ANTHROPIC_API_KEY（否則 Claude 會回 500）
3) 若 main.py 啟用金鑰檢查(ENABLE_CHECK_APIKEY=true)，需提供白名單內的 Bearer Token
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import requests


OPENLLM_GATEWAY_BASE_URL = os.getenv("OPENLLM_GATEWAY_BASE_URL", "http://localhost:8000/v1").rstrip("/")
OPENLLM_GATEWAY_API_KEY = os.getenv("OPENLLM_GATEWAY_API_KEY", "sk-Ko04aszCTvwUc3QZTubDbOkJK30UEQlZmUjgC5g2Z0X6g3cj")

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "xiaomi/mimo-v2.5-pro")
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "120"))


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENLLM_GATEWAY_API_KEY}",
        "Content-Type": "application/json",
    }


def _gateway_url(path: str) -> str:
    return f"{OPENLLM_GATEWAY_BASE_URL}{path}"


def chat_completion_non_streaming() -> None:
    print(f"--- 非串流 Claude 測試 (model={CHAT_MODEL_NAME}) ---")
    payload: Dict[str, Any] = {
        "model": CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": "用一句話解釋什麼是 Open LLM Gateway。"},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": False,
    }

    resp = requests.post(
        _gateway_url("/chat/completions"),
        headers=_headers(),
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )

    if resp.status_code >= 400:
        print(f"HTTP {resp.status_code}")
        try:
            print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
        except Exception:
            print(resp.text)
        return

    data = resp.json()
    message = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    usage = data.get("usage")
    print("Assistant:", message)
    if usage:
        print("Usage:", usage)
    print()


def _iter_sse_lines(response: requests.Response) -> Iterable[str]:
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        yield raw


def chat_completion_streaming() -> None:
    print(f"--- 串流 Claude 測試 (model={CHAT_MODEL_NAME}) ---")
    payload: Dict[str, Any] = {
        "model": CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a poetic assistant, skilled in crafting short verses."},
            {"role": "user", "content": "寫一首四行短詩，主題是「月光下的安靜街道」。"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,
    }

    with requests.post(
        _gateway_url("/chat/completions"),
        headers=_headers(),
        json=payload,
        stream=True,
        timeout=TIMEOUT_SECONDS,
    ) as resp:
        if resp.status_code >= 400:
            print(f"HTTP {resp.status_code}")
            try:
                print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
            except Exception:
                print(resp.text)
            return

        full_text = ""
        for line in _iter_sse_lines(resp):
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(chunk, dict) and "error" in chunk:
                print("\nError chunk:")
                print(json.dumps(chunk, ensure_ascii=False, indent=2))
                break

            delta = (((chunk.get("choices") or [{}])[0].get("delta") or {}) if isinstance(chunk, dict) else {})
            piece: Optional[str] = delta.get("content") if isinstance(delta, dict) else None
            if piece:
                print(piece, end="", flush=True)
                full_text += piece

        print("\n\n--- 串流結束 ---")
        if full_text:
            print("Full text:", full_text)
    print()


def main() -> None:
    print("Open LLM Gateway - Anthropic(Claude) 範例腳本")
    print(f"Gateway: {OPENLLM_GATEWAY_BASE_URL}")
    print(f"Model: {CHAT_MODEL_NAME}")
    print("-" * 50)
    chat_completion_non_streaming()
    chat_completion_streaming()


if __name__ == "__main__":
    main()

