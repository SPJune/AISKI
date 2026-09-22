#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typecast voice filter script (corrected)
---------------------------------------
- Endpoint: GET https://api.typecast.ai/v1/voices
- Auth: X-API-KEY: <your_api_key>
- Pagination: Not required (returns full list); optional ?model=ssfm-v30

Filters voices by:
  1) Korean (standard) support
  2) Excluding content categories: game/anime, music/entertainment
  3) Duration parameter supported (heuristic)

Outputs:
  - CSV (name, voice_id, language, categories)
  - JSON (full filtered records)
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

import requests  # pip install requests

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Typecast voices (Korean standard, exclude game/music-entertainment, supports duration).")
    p.add_argument("--api-key", default=os.environ.get("TYPECAST_API_KEY") or os.environ.get("TYPECAST_API_TOKEN"),
                   help="Typecast API key; or set env TYPECAST_API_KEY / TYPECAST_API_TOKEN")
    p.add_argument("--base-url", default="https://api.typecast.ai", help="Base URL (default: https://api.typecast.ai)")
    p.add_argument("--model", default=None, help="Optional model filter, e.g., ssfm-v30")
    p.add_argument("--out-csv", default="data/reference/typecast_ko_standard_duration.csv", help="Output CSV filename")
    p.add_argument("--out-json", default="data/reference/typecast_ko_standard_duration.json", help="Output JSON filename")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

def headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": api_key,
        "Accept": "application/json",
    }

def fetch_voices(base_url: str, api_key: str, model: Optional[str], timeout: float, verbose: bool) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/voices"
    params = {}
    if model:
        params["model"] = model
    if verbose:
        print(f"[i] GET {url} params={params or None}")
    r = requests.get(url, headers=headers(api_key), params=params or None, timeout=timeout)
    if r.status_code == 401:
        print("ERROR: Unauthorized (401). Check X-API-KEY value and permissions.", file=sys.stderr)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
        return data["voices"]
    # fallback: find first list of dicts
    for k, v in (data.items() if isinstance(data, dict) else []):
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise RuntimeError("Unexpected response shape from /v1/voices")

# -------- schema helpers --------

def normalize_voice_id(v: Dict[str, Any]) -> Optional[str]:
    for k in ("voice_id", "id", "voiceId", "uuid", "key"):
        val = v.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

def normalize_voice_name(v: Dict[str, Any]) -> Optional[str]:
    for k in ("voice_name", "name", "title", "display_name", "label"):
        val = v.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

def get_languages(v: Dict[str, Any]) -> Set[str]:
    langs: Set[str] = set()
    def add_val(x):
        if isinstance(x, str):
            langs.add(x.lower())
        elif isinstance(x, list):
            for y in x:
                if isinstance(y, str):
                    langs.add(y.lower())
    for k in ("language", "lang", "locale", "locales", "languages"):
        if k in v:
            add_val(v[k])
    for parent in ("metadata", "attributes", "profile"):
        if isinstance(v.get(parent), dict):
            for k in ("language", "lang", "locale", "locales", "languages"):
                if k in v[parent]:
                    add_val(v[parent][k])
    return langs

def get_tags_or_categories(v: Dict[str, Any]) -> Set[str]:
    values: Set[str] = set()
    keys = ("tags", "categories", "category", "content_categories", "content", "genres", "use_cases")
    parents = ("metadata", "attributes", "profile")
    def add(x):
        if isinstance(x, list):
            for item in x:
                if isinstance(item, str):
                    values.add(item.strip().lower())
        elif isinstance(x, str):
            values.add(x.strip().lower())
    for k in keys:
        if k in v:
            add(v[k])
    for p in parents:
        if isinstance(v.get(p), dict):
            for k in keys:
                if k in v[p]:
                    add(v[p][k])
    return values

EXCLUDED_KO_CONTENT = {
    "게임/애니", "게임", "애니", "애니메이션",
    "music/entertainment", "music", "entertainment",
    "음악", "엔터테인먼트", "음악/엔터테인먼트",
}
ALLOWED_KO_CONTENT = {
    "다큐/리뷰", "다큐", "리뷰",
    "낭독/오디오북", "낭독", "오디오북",
    "기자/아나운서", "기자", "아나운서",
    "라디오/팟캐스트", "라디오", "팟캐스트",
    "광고/이벤트", "광고", "이벤트",
    "교육/강의", "교육", "강의",
    "안내음성/ars", "안내음성", "ars",
}
ALLOWED_EN_CONTENT = {
    "documentary", "review", "docu", "narration", "audiobook",
    "news", "anchor", "announcer", "journalist",
    "radio", "podcast",
    "advertising", "ad", "event",
    "education", "lecture", "course",
    "ivr", "ars", "announcement", "prompt", "guide", "navigation", "system",
}
EXCLUDED_EN_CONTENT = {"game", "gaming", "anime", "animation", "music", "entertainment"}

def supports_duration(v: Dict[str, Any]) -> bool:
    for key in ("supports_duration", "duration_supported", "can_set_duration"):
        if isinstance(v.get(key), bool) and v[key]:
            return True
    for key in ("features", "capabilities", "options", "supported_parameters"):
        val = v.get(key)
        if isinstance(val, list):
            vals = {str(x).lower() for x in val}
            if any("duration" in s for s in vals):
                return True
    for parent in ("metadata", "attributes", "profile", "tts"):
        sub = v.get(parent)
        if isinstance(sub, dict):
            for key in ("supports_duration", "duration_supported", "can_set_duration"):
                if isinstance(sub.get(key), bool) and sub[key]:
                    return True
            for key in ("features", "capabilities", "options", "supported_parameters"):
                val = sub.get(key)
                if isinstance(val, list):
                    vals = {str(x).lower() for x in val}
                    if any("duration" in s for s in vals):
                        return True
    for key in ("max_seconds", "duration_range", "duration"):
        val = v.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return True
        if isinstance(val, dict) and any(k in val for k in ("min", "max")):
            return True
    return False

def is_korean_standard(langs: Set[str], tags: Set[str], voice: Dict[str, Any]) -> bool:
    has_ko = any(s.startswith("ko") for s in langs) or "korean" in langs
    if not has_ko:
        return False
    for dk in ("dialect", "accent", "style"):
        val = voice.get(dk)
        if isinstance(val, str):
            lv = val.lower()
            if "standard" in lv or "표준" in lv or "표준어" in lv:
                return True
            if any(x in lv for x in ("경상", "전라", "충청", "제주", "사투리", "방언", "dialect", "busan", "gyeongsang", "jeju")):
                return False
    if any(x in tags for x in ("표준", "표준어", "standard")):
        return True
    return True

def content_ok(tags: Set[str]) -> bool:
    if any(t in EXCLUDED_KO_CONTENT for t in tags) or any(t in EXCLUDED_EN_CONTENT for t in tags):
        return False
    if any(t in ALLOWED_KO_CONTENT for t in tags) or any(t in ALLOWED_EN_CONTENT for t in tags):
        return True
    return False

def filter_voices(voices: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for v in voices:
        langs = get_languages(v)
        tags = get_tags_or_categories(v)
        if not is_korean_standard(langs, tags, v):
            continue
        if not content_ok(tags):
            continue
        if not supports_duration(v):
            continue
        out.append(v)
    if verbose:
        print(f"[i] Filtered voices: {len(out)}")
    return out

def save_csv(voices: List[Dict[str, Any]], path: str) -> None:
    cols = ["name", "voice_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for v in voices:
            name = normalize_voice_name(v) or ""
            vid = normalize_voice_id(v) or ""
            w.writerow([name, vid])

def save_json(voices: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    if not args.api_key:
        print("ERROR: Provide --api-key or set TYPECAST_API_KEY / TYPECAST_API_TOKEN", file=sys.stderr)
        sys.exit(2)
    voices = fetch_voices(args.base_url, args.api_key, args.model, args.timeout, args.verbose)
    save_csv(voices, args.out_csv)
    save_json(voices, args.out_json)
    print(f"[✓] Saved CSV: {args.out_csv}")
    print(f"[✓] Saved JSON: {args.out_json}")
    print(f"[i] Total filtered voices: {len(voices)}")

if __name__ == "__main__":
    main()
