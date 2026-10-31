#!/usr/bin/env python3
"""Generate Piper "next stop" clips for bustimes.org routes.

Run by the GitHub workflow when a route is picked in the app:

    python scripts/generate.py --service 12345[,67890]

For each route (bustimes.org service id) it fetches the route's stops,
compares them with audio/services/<id>.json, and only synthesises stops
that were added (or whose spoken name / voice changed). Stops removed
from every route are deleted. Nothing is written if nothing changed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
STOPS_DIR = AUDIO / "stops"
SERVICES_DIR = AUDIO / "services"
COMMON_DIR = AUDIO / "common"
INDEX_FILE = STOPS_DIR / "index.json"
API = "https://bustimes.org"
UA = "next-stop-announcements (GitHub Actions; personal use)"

COMMON_PHRASES = {
    "next_stop": "Next stop:",
    "final_stop": "This is the final stop.",
    "terminates": "This bus terminates here.",
}


# ---------------------------------------------------------------- names
# Kept in step with cleanName()/speakable() in index.html
def clean_name(raw: str) -> str:
    n = str(raw or "")
    n = re.sub(r"\b(opp(osite)?|adj(acent)?)\b\.?", " ", n, flags=re.I)
    n = re.sub(r"\bstand\s*[A-Z]?\d{0,3}[A-Z]?\b", " ", n, flags=re.I)
    n = re.sub(r"\(\s*[,\-–/]*\s*\)", " ", n)
    n = re.sub(r"\s*[,\-–/]\s*(?=\)|$)", "", n)
    n = re.sub(r"\(\s*[,\-–/]\s*", "(", n)
    n = re.sub(r"\s{2,}", " ", n)
    n = re.sub(r"^[\s,\-–/]+|[\s,\-–/]+$", "", n).strip()
    return n or str(raw or "Stop").strip()


ABBREVIATIONS = [
    (r"^St\b\.?", "Saint", 0), (r"\bStn\b", "Station", re.I), (r"\bRd\b", "Road", 0), (r"\bSt\b(?=\s|$)", "Street", 0),
    (r"\bAve?\b", "Avenue", 0), (r"\bLn\b", "Lane", 0), (r"\bSq\b", "Square", 0),
    (r"\bHosp\b", "Hospital", re.I), (r"\bCres\b", "Crescent", 0), (r"\bDr\b(?=\s|$)", "Drive", 0),
    (r"\bGdns\b", "Gardens", 0), (r"\bPk\b", "Park", 0), (r"\bCl\b(?=\s|$)", "Close", 0),
    (r"\bCtr\b", "Centre", 0), (r"\bInt\b", "Interchange", 0), (r"&", " and ", 0),
]


def speakable(raw: str) -> str:
    n = re.sub(r"[()]", " ", clean_name(raw))
    for pattern, repl, flags in ABBREVIATIONS:
        n = re.sub(pattern, repl, n, flags=flags)
    return re.sub(r"\s{2,}", " ", n).strip()


# ---------------------------------------------------------------- bustimes
def get_json(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"  retry {url}: {exc}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def fetch_service_stops(service_id: str) -> dict[str, str]:
    """{atco_code: display name} for every stop on the route."""
    stops: dict[str, str] = {}
    url = f"{API}/api/stops/?service={service_id}"
    while url:
        data = get_json(url)
        for s in data.get("results", []):
            code = s.get("atco_code")
            if code:
                stops[code] = s.get("name") or s.get("common_name") or s.get("long_name") or code
        url = data.get("next")
    return stops


def fetch_service_info(service_id: str) -> dict:
    try:
        return get_json(f"{API}/api/services/{service_id}/")
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- audio
class Synth:
    def __init__(self, voice: str, length_scale: float | None, voice_dir: Path):
        from piper import PiperVoice, SynthesisConfig  # imported here so --dry-run works without piper

        model = voice_dir / f"{voice}.onnx"
        if not model.exists():
            subprocess.run([sys.executable, "-m", "piper.download_voices", voice, "--download-dir", str(voice_dir)], check=True)
        self.voice = PiperVoice.load(str(model))
        self.config = SynthesisConfig(length_scale=length_scale)

    def mp3(self, text: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            with wave.open(str(wav_path), "wb") as wf:
                self.voice.synthesize_wav(text, wf, syn_config=self.config)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                 "-af", "silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                 "-ac", "1", "-ar", "22050", "-b:a", "48k", str(out)],
                check=True,
            )
        finally:
            wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------- main
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run(service_ids: list[str], dry_run: bool = False, synth_factory=None) -> list[str]:
    config = load_json(ROOT / "config.json", {})
    voice = config.get("voice", "en_GB-cori-high")
    length_scale = config.get("length_scale")
    overrides = load_json(ROOT / "overrides.json", {})
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
    index: dict = load_json(INDEX_FILE, {})
    summary: list[str] = []
    to_make: dict[Path, str] = {}

    # common phrases, regenerated if the voice changes
    common_meta = load_json(COMMON_DIR / "index.json", {})
    for key, text in COMMON_PHRASES.items():
        path = COMMON_DIR / f"{key}.mp3"
        if not path.exists() or common_meta.get(key) != {"text": text, "voice": voice}:
            to_make[path] = text
            common_meta[key] = {"text": text, "voice": voice}

    for sid in service_ids:
        print(f"Route {sid}: fetching stops…")
        stops = fetch_service_stops(sid)
        if not stops:
            summary.append(f"- Route {sid}: no stops returned, left unchanged")
            continue
        manifest_path = SERVICES_DIR / f"{sid}.json"
        old = load_json(manifest_path, {})
        old_codes = set(old.get("stops", []))
        new_codes = set(stops)
        added, removed = sorted(new_codes - old_codes), sorted(old_codes - new_codes)

        for code, name in stops.items():
            text = overrides.get(code) or speakable(name)
            entry = {"text": text, "voice": voice}
            if index.get(code) != entry or not (STOPS_DIR / f"{code}.mp3").exists():
                to_make[STOPS_DIR / f"{code}.mp3"] = text
                index[code] = entry

        if added or removed or not manifest_path.exists():
            info = fetch_service_info(sid)
            write_json(manifest_path, {
                "service_id": int(sid) if sid.isdigit() else sid,
                "line_name": info.get("line_name") or old.get("line_name"),
                "description": info.get("description") or old.get("description"),
                "stops": sorted(new_codes),
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        summary.append(f"- Route {sid}: {len(new_codes)} stops, {len(added)} added, {len(removed)} removed")

    # delete clips no route uses any more
    used = set()
    for mf in SERVICES_DIR.glob("*.json"):
        used.update(load_json(mf, {}).get("stops", []))
    orphans = [c for c in list(index) if c not in used]
    for code in orphans:
        index.pop(code, None)
        to_make.pop(STOPS_DIR / f"{code}.mp3", None)
        if not dry_run:
            (STOPS_DIR / f"{code}.mp3").unlink(missing_ok=True)
    if orphans:
        summary.append(f"- Deleted {len(orphans)} clips no longer on any route")

    summary.append(f"- Clips to generate: {len(to_make)} (voice {voice})")
    if dry_run:
        for path, text in sorted(to_make.items()):
            print(f"  would make {path.relative_to(ROOT)}: {text!r}")
    elif to_make:
        synth = (synth_factory or Synth)(voice, length_scale, ROOT / ".voices")
        for i, (path, text) in enumerate(sorted(to_make.items()), 1):
            synth.mp3(text, path)
            if i % 25 == 0 or i == len(to_make):
                print(f"  {i}/{len(to_make)} clips")

    if not dry_run:
        write_json(INDEX_FILE, index)
        write_json(COMMON_DIR / "index.json", common_meta)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", required=True, help="bustimes.org service id(s), comma separated")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ids = [s.strip() for s in args.service.split(",") if s.strip()]
    if not ids or not all(re.fullmatch(r"\d{1,10}", s) for s in ids):
        sys.exit("--service must be numeric bustimes.org service ids, e.g. 12345 or 12345,67890")
    summary = run(ids, dry_run=args.dry_run)
    text = "### Stop announcements\n" + "\n".join(summary) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
