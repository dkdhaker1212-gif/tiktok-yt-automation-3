"""AI-generated YouTube SEO: title, description (with hashtags), tags.

Uses the Anthropic API (Claude Haiku -- cheap) when ANTHROPIC_API_KEY is set.
Falls back to a deterministic template so a missing key never drops a slot.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass

_MODEL = os.environ.get("SEO_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class Seo:
    title: str
    description: str
    tags: list[str]
    thumb_hook: str = ""


def _norm(t: str) -> str:
    t = (t or "").lower().replace("#shorts", "")
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_dupe(title: str, recent) -> bool:
    n = _norm(title)
    if not n:
        return True
    for r in recent or []:
        rn = _norm(r)
        if not rn:
            continue
        if n == rn:
            return True
        a, b = set(n.split()), set(rn.split())
        if a and b and len(a & b) / max(len(a), len(b)) >= 0.85:
            return True
    return False


_RECAP_MUT = [
    "You Missed This", "Watch It Twice", "The Twist Hits Different",
    "It All Connects", "Nobody Talks About This", "The Detail Everyone Skips",
]


def _mutate(title: str, recent) -> str:
    base = title.split("|")[0].strip(" -|")
    used = {_norm(r) for r in (recent or [])}
    for h in _RECAP_MUT:
        cand = f"{base} - {h}"
        if _norm(cand) not in used:
            return cand[:98]
    return f"{base} (v{abs(hash(base)) % 999})"[:98]


_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
         "this", "that", "it", "is", "are", "my", "your", "you", "we", "so"}


def _keywords(text: str, n: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']+", (text or "").lower())
    seen: list[str] = []
    for w in words:
        if len(w) > 2 and w not in _STOP and w not in seen:
            seen.append(w)
        if len(seen) >= n:
            break
    return seen


_RECAP_TITLES = [
    "This Movie Has One of the Craziest Plot Twists Ever | Full Recap",
    "The Ending of This Movie Left Everyone Speechless | Movie Recap",
    "You Won't Believe How This Movie Ends | Full Movie Recap",
    "This Thriller Keeps You Guessing Until the Last Minute | Recap",
    "Nobody Saw This Twist Coming | Full Movie Recap & Ending Explained",
    "This Movie Starts Normal Then Turns Into a Nightmare | Recap",
    "The Whole Movie Was a Lie | Movie Recap & Ending Explained",
    "This Is Why Everyone Is Talking About This Movie | Full Recap",
]
_US_HASHTAGS = ["#movierecap", "#movierecaps", "#moviereview", "#endingexplained",
                "#movies", "#hollywood", "#film", "#movietok", "#netflix",
                "#moviescene", "#thriller", "#fyp"]


def _fallback(caption: str, tiktok_tags: list[str], base_tags: list[str],
              is_short: bool) -> Seo:
    cap = (caption or "").strip().replace("\n", " ")
    cap_clean = re.sub(r"#\S+", "", cap).strip(" .,-|")   # drop hashtags
    kws = _keywords(cap)

    # caption is just hashtags / too thin -> use a rotating recap hook
    if len(cap_clean) < 12:
        seed = sum(ord(c) for c in (caption or "x"))
        title = _RECAP_TITLES[seed % len(_RECAP_TITLES)]
    else:
        title = cap_clean[:88].rstrip(" .,-")
    if is_short and "#shorts" not in title.lower():
        title = (title[:88] + " #Shorts").strip()

    extra = ["shorts", "viral", "trending", "fyp"] if is_short else \
            ["viral", "trending", "fyp", "usa"]
    tags = list(dict.fromkeys(
        [*base_tags, *[t.lower() for t in tiktok_tags], *kws, *extra]
    ))[:25]

    hs = (["#shorts"] if is_short else []) + _US_HASHTAGS
    hashtags = " ".join(dict.fromkeys(hs[:8]))
    desc = "\n".join(filter(None, [
        title.replace(" #Shorts", ""),
        "",
        "Full movie recap and ending explained. Subscribe for a new recap every day.",
        "",
        hashtags,
    ])).strip()
    return Seo(title[:100], desc[:4900], tags)


_PROMPT = """You are a YouTube Shorts SEO expert. Given a TikTok video's caption and \
its original hashtags, produce optimized YouTube metadata.

Return ONLY minified JSON with keys: "title", "description", "tags".
Rules:
- title: <= 90 chars, punchy, curiosity-driven, front-load the keyword, natural \
English, no clickbait lies, end with " #Shorts" if is_short is true.
- description: 2-4 short lines. Line 1 = a hook. Then a blank line. Then 4-6 \
relevant hashtags on one line. No links. <= 400 chars.
- tags: 15-20 lowercase search phrases, most specific first, no "#".
Caption: {caption}
Original hashtags: {hashtags}
is_short: {is_short}
"""


FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

_GEMINI_PROMPT = """You are given the audio of a faceless "movie recap" video.
Identify the EXACT movie being recapped (name + release year). Base it on plot,
character names, and dialogue you hear.

Return ONLY minified JSON:
{"movie","title","description","tags","hashtags","thumb_hook"}
- movie: "Name (Year)" or null if you truly cannot tell.
- title: <= 90 chars. Front-load the movie name. Strong curiosity / clickbait
  energy but honest, no ALL-CAPS spam. e.g.
  "Everyone Missed This Detail in <Movie> | Full Recap".
  It MUST be clearly different in wording AND structure from every title in
  ALREADY_USED below - different hook, not just a swapped word.
- description: 180-320 words. Line 1 = a hook. Then name the movie, year, genre,
  and lead actors if you can. One spoiler-free setup paragraph about the premise.
  End with "New movie recaps every day - subscribe." Then a blank line and 6-8
  US-style hashtags.
- tags: 20 lowercase search phrases, no '#'. Include the movie name, lead actors,
  genre, and "movie recap"/"ending explained"/"full movie recap" variants.
- hashtags: 8 strings starting with '#', US style, include #movierecap.
- thumb_hook: 2-4 punchy words for the thumbnail, no emojis, no hashtags.
Original caption/hashtags from the source: {caption}
ALREADY_USED (do NOT repeat or lightly reword any of these):
{recent}
"""


def _extract_audio(media_path, out_m4a):
    # AAC/m4a: always available in ffmpeg (mp3 encoder is often missing in
    # static builds). Gemini mime type: audio/mp4.
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "aac", "-b:a", "64k", out_m4a],
            capture_output=True, text=True, timeout=300)
        ok = os.path.isfile(out_m4a) and os.path.getsize(out_m4a) > 1000
        if not ok:
            print(f"[seo] audio extract failed rc={r.returncode}: "
                  f"{(r.stderr or '')[-300:]}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"[seo] audio extract failed: {exc}")
        return False


def _gemini(media_path, caption, base_tags, recent_titles=None):
    import base64 as _b64
    import urllib.request

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not (key and media_path and os.path.isfile(media_path)):
        return None
    aud = media_path + ".seo.m4a"
    try:
        if not _extract_audio(media_path, aud):
            return None
        if os.path.getsize(aud) > 18 * 1024 * 1024:      # inline cap ~20MB
            print("[seo] audio too big for inline Gemini; skipping")
            return None
        audio_b64 = _b64.b64encode(open(aud, "rb").read()).decode()
        recent = "\n".join(f"- {t[:90]}" for t in (recent_titles or [])[:30]) or "(none)"
        body = json.dumps({
            "contents": [{"parts": [
                {"text": _GEMINI_PROMPT
                 .replace("{caption}", (caption or "(none)")[:300])
                 .replace("{recent}", recent)},
                {"inline_data": {"mime_type": "audio/mp4", "data": audio_b64}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "temperature": 0.5, "maxOutputTokens": 2600},
        }).encode()
        import time
        import urllib.error
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{_GEMINI_MODEL}:generateContent?key={key}")
        resp = None
        for attempt, wait in enumerate(([0, 8, 20, 45]), start=1):
            if wait:
                time.sleep(wait)
            req = urllib.request.Request(url, data=body,
                                        headers={"Content-Type": "application/json"})
            try:
                resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
                break
            except urllib.error.HTTPError as he:
                if he.code in (429, 500, 502, 503) and attempt < 4:
                    print(f"[seo] Gemini {he.code}, retry {attempt}")
                    continue
                raise
        if resp is None:
            return None
        parts = resp["candidates"][0]["content"]["parts"]
        raw = "".join(p["text"] for p in parts if isinstance(p.get("text"), str))
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        i, j = raw.find("{"), raw.rfind("}")          # tolerate stray prose
        data = json.loads(raw[i:j + 1] if i != -1 and j != -1 else raw)
        title = str(data.get("title", "")).strip()[:100]
        desc = str(data.get("description", "")).strip()[:4900]
        tags = [str(t).strip().lower().lstrip("#") for t in data.get("tags", []) if t]
        hashtags = [str(h).strip() for h in data.get("hashtags", []) if h]
        hook = str(data.get("thumb_hook", "")).strip()[:40]
        if hashtags and "#" not in desc:
            desc = (desc + "\n\n" + " ".join(hashtags[:8])).strip()
        tags = list(dict.fromkeys([*tags, *base_tags]))[:30]
        if not title or not tags:
            return None
        print(f"[seo] Gemini metadata OK (movie={data.get('movie')!r})")
        return Seo(title, desc, tags, thumb_hook=hook)
    except Exception as exc:  # noqa: BLE001
        print(f"[seo] Gemini failed ({exc}); falling back")
        return None
    finally:
        try:
            os.remove(aud)
        except OSError:
            pass


def generate(caption: str, tiktok_tags: list[str], base_tags: list[str],
             is_short: bool, media_path: str | None = None) -> Seo:
    # preferred: analyse the actual audio with Gemini (identifies the movie)
    if os.environ.get("GEMINI_API_KEY", "").strip() and media_path:
        g = _gemini(media_path, caption, base_tags)
        if g:
            if is_short and "#shorts" not in g.title.lower() and len(g.title) <= 91:
                g.title = (g.title + " #Shorts").strip()
            return g

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("[seo] no GEMINI/ANTHROPIC key usable; template fallback")
        return _fallback(caption, tiktok_tags, base_tags, is_short)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=700,
            messages=[{
                "role": "user",
                "content": _PROMPT.format(
                    caption=(caption or "")[:800],
                    hashtags=", ".join(tiktok_tags[:20]) or "(none)",
                    is_short=str(bool(is_short)).lower(),
                ),
            }],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        data = json.loads(raw)
        title = str(data.get("title", "")).strip()[:100]
        desc = str(data.get("description", "")).strip()[:4900]
        tags = [str(t).strip().lower().lstrip("#") for t in data.get("tags", []) if t]
        tags = list(dict.fromkeys([*base_tags, *tags]))[:25]
        if not title or not tags:
            raise ValueError("model returned empty title/tags")
        if is_short and "#shorts" not in title.lower() and len(title) <= 91:
            title = (title + " #Shorts").strip()
        print(f"[seo] AI metadata OK ({_MODEL})")
        return Seo(title, desc, tags)
    except Exception as exc:  # noqa: BLE001
        print(f"[seo] AI generation failed ({exc}); using template fallback")
        return _fallback(caption, tiktok_tags, base_tags, is_short)
