#!/usr/bin/env python3
"""Probe Gemini model availability and multimodal generateContent support.

This script:
- Loads .env (GEMINI_API_KEYS/GEMINI_API_KEY)
- Lists models visible to your key
- Probes candidate models by calling generate_content with a tiny image + short prompt
- Prints a recommended GEMINI_MODELS JSON list (without the 'models/' prefix)

Usage examples:
  python scripts/probe_gemini_models.py
  python scripts/probe_gemini_models.py --only-env-models
  python scripts/probe_gemini_models.py --limit 8 --sleep 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

from google import genai
from google.genai import types


EXCLUDE_MARKERS = (
    "embedding",
    "native-audio",
    "tts",
    "robotics",
    "computer-use",
)


def _load_first_api_key_from_env() -> str:
    keys_raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY")
    api_key: Optional[str] = None

    if keys_raw:
        try:
            parsed = json.loads(keys_raw)
            if isinstance(parsed, list) and parsed:
                api_key = str(parsed[0]).strip()
            elif isinstance(parsed, str) and parsed:
                api_key = parsed.strip()
        except Exception:
            api_key = keys_raw.strip()

    if not api_key:
        raise SystemExit("No API key found. Set GEMINI_API_KEYS (JSON list) or GEMINI_API_KEY in .env")

    return api_key


def _strip_models_prefix(name: str) -> str:
    return name.split("models/", 1)[1] if name.startswith("models/") else name


def _model_score(name: str) -> int:
    n = name.lower()
    # lower is better
    s = 100
    if "-image" in n:
        s -= 40
    if "flash-lite" in n:
        s -= 30
    if "flash" in n:
        s -= 20
    if "pro" in n:
        s -= 10
    if "preview" in n:
        s += 10
    if "exp" in n:
        s += 20
    return s


def _stable_rank(name: str) -> Tuple[bool, int, int, str]:
    n = name.lower()
    unstable = ("preview" in n) or ("exp" in n)

    # tier preference: lite -> flash -> pro
    tier = 3
    if "flash-lite" in n:
        tier = 0
    elif "flash" in n:
        tier = 1
    elif "pro" in n:
        tier = 2

    # prefer non "-image" unless explicitly needed
    image_bonus = 0 if "-image" in n else 1
    return (unstable, tier, image_bonus, n)


def _env_models_list() -> List[str]:
    raw = os.environ.get("GEMINI_MODELS")
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(m).strip() for m in parsed if str(m).strip()]
    except Exception:
        return [m.strip() for m in raw.split(",") if m.strip()]

    return []


def _pick_candidates(
    visible_models: Iterable[str],
    *,
    only_env_models: bool,
    limit: int,
    include_preview: bool,
) -> List[str]:
    visible = list(visible_models)

    if only_env_models:
        env_models = _env_models_list()
        if not env_models:
            raise SystemExit("--only-env-models was set but GEMINI_MODELS is missing/empty in .env")

        # normalize: allow env list without 'models/' prefix
        env_full = set()
        for m in env_models:
            env_full.add(m)
            env_full.add(f"models/{m}")

        candidates = [m for m in visible if m in env_full]
        return candidates

    candidates = []
    for m in visible:
        ml = m.lower()
        if "gemini" not in ml:
            continue
        if any(x in ml for x in EXCLUDE_MARKERS):
            continue
        if (not include_preview) and ("preview" in ml or "exp" in ml):
            continue
        candidates.append(m)

    candidates = sorted(candidates, key=lambda n: (_model_score(n), n))
    return candidates[:limit]


def probe_multimodal(
    client: genai.Client,
    model_names: List[str],
    *,
    sleep_seconds: float,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    prompt = "Reply with only: OK"

    config = types.GenerateContentConfig(
        max_output_tokens=16,
        temperature=0.0,
    )

    supported: List[str] = []
    failed: List[Tuple[str, str]] = []

    for idx, model_name in enumerate(model_names, 1):
        if sleep_seconds > 0 and idx > 1:
            time.sleep(sleep_seconds)

        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[img, prompt],
                config=config,
            )
            text = (getattr(resp, "text", "") or "").strip()
            supported.append(model_name)
            print(f"[{idx:02d}/{len(model_names)}] OK   {model_name} -> {text[:30]!r}")
        except Exception as e:
            msg = str(e)
            failed.append((model_name, msg))
            print(f"[{idx:02d}/{len(model_names)}] FAIL {model_name} -> {msg[:120]}")

    return supported, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="List and probe Gemini models for multimodal generateContent")
    parser.add_argument("--limit", type=int, default=12, help="How many candidates to probe (when not using --only-env-models)")
    parser.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between probes")
    parser.add_argument(
        "--only-env-models",
        action="store_true",
        help="Probe only models listed in GEMINI_MODELS from .env",
    )
    parser.add_argument(
        "--include-preview",
        action="store_true",
        help="Include preview/exp models in automatic candidate selection",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(".env"), override=False)
    api_key = _load_first_api_key_from_env()

    client = genai.Client(api_key=api_key)

    visible_models = [getattr(m, "name", str(m)) for m in client.models.list()]
    print(f"Visible models returned: {len(visible_models)}")

    candidates = _pick_candidates(
        visible_models,
        only_env_models=args.only_env_models,
        limit=max(1, args.limit),
        include_preview=args.include_preview,
    )

    if not candidates:
        raise SystemExit("No candidates found to probe.")

    print(f"Probing candidates: {len(candidates)}")
    supported, failed = probe_multimodal(client, candidates, sleep_seconds=max(0.0, args.sleep))

    print("\n--- Supported (multimodal generateContent) ---")
    for m in supported:
        print(m)

    print("\n--- Failed (first 10) ---")
    for m, err in failed[:10]:
        print(f"{m} :: {err[:160]}")

    recommended = sorted(supported, key=_stable_rank)
    recommended_env = [_strip_models_prefix(m) for m in recommended]

    print("\n--- Recommended GEMINI_MODELS (JSON) ---")
    print(json.dumps(recommended_env, ensure_ascii=False))


if __name__ == "__main__":
    main()
