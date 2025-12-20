#!/usr/bin/env python3
# text_tools.py
"""
Text labeling tool for multi-speaker TTS:
- Robust quote/dialogue extraction (multiple quotes per line)
- Speaker candidate heuristics (X said / said X patterns)
- Emotion heuristic scoring as fallback
- LLM labeling prompt with id-aligned strict JSON

You can integrate this as an agent tool:
- label_text(text=..., roster=..., use_llm=True) -> list of labeled units
- label_text_from_file(file_path=...) -> list of labeled units
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import regex as re  # keep your regex dependency


EMOTIONS = ["neutral","happy","sad","angry","fear","surprise","disgust","calm","excited"]
GENDERS  = ["male","female","unknown"]

QUOTE_OPEN = "“『「"
QUOTE_CLOSE = "”』」"

# 说话动词：用来找 speaker attribution（你可以继续扩充）
SAY_VERBS = [
    "说","问","答","道","喊","叫","吼","嚷","嘀咕","嘟囔","喃喃","低声说","轻声说","冷笑","笑道","哭道","骂道",
    "提醒","解释","劝","命令","宣布","咆哮","尖叫"
]
SAY_VERB_RE = re.compile("|".join(map(re.escape, sorted(SAY_VERBS, key=len, reverse=True))))

# “某某说/问/道/喊道：” 这类
SPEAKER_BEFORE_RE = re.compile(
    rf"(?P<spk>[\p{{Han}}]{{1,6}})(?:[^\p{{Han}}]{{0,3}})?(?:{SAY_VERB_RE.pattern})(?:[：:，, ]|$)"
)
# “……”，某某说。 这类
SPEAKER_AFTER_RE = re.compile(
    rf"(?:[，,。！!？?]\s*)?(?P<spk>[\p{{Han}}]{{1,6}})(?:[^\p{{Han}}]{{0,3}})?(?:{SAY_VERB_RE.pattern})"
)

# 情绪启发：改成“多命中评分”而不是命中第一个就停
EMO_SCORES: List[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"(怒|生气|火冒|拍桌|咆哮|吼|嚷|骂|瞪)"), "angry", 2.0),
    (re.compile(r"(哭|泪|哽咽|抽泣|委屈|难过|心酸|悲伤)"), "sad", 2.0),
    (re.compile(r"(害怕|恐惧|发抖|颤|惊恐|尖叫|躲)"), "fear", 2.0),
    (re.compile(r"(惊|愣|突然|骤然|一下子|震惊)"), "surprise", 1.5),
    (re.compile(r"(笑|开心|高兴|愉快|兴奋|雀跃)"), "happy", 1.5),
    (re.compile(r"(平静|温和|镇定|缓缓|轻轻|柔声)"), "calm", 1.2),
    (re.compile(r"(恶心|厌恶|嫌弃|作呕)"), "disgust", 1.5),
    (re.compile(r"(激动|亢奋|热血|振奋)"), "excited", 1.2),
]

# 旁白/描写更靠谱的线索：场景、动作、环境声音（不要再把“他/她”当旁白线索）
NARRATION_HINT_RE = re.compile(
    r"(四周|夜色|阳光|风|雨|雷|空气|街道|屋里|门外|脚步声|沉默|安静|火锅|咕嘟|钟声|灯光|墙|桌|椅|走|站|坐|靠|转身|抬手|握|抱|躲|快步)"
)

# 用于“去掉括号/舞台提示”之类
BRACKET_RE = re.compile(r"[\(\（\[\【].*?[\)\）\]\】]")


def split_sentences_quote_aware(text: str) -> List[str]:
    """
    Split text into sentences by 。！？；… and newlines, but do NOT cut inside Chinese quotes.
    """
    s = text.strip()
    if not s:
        return []

    out: List[str] = []
    buf: List[str] = []
    stack = 0  # quote nesting level (simple)

    def flush():
        nonlocal buf
        sent = "".join(buf).strip()
        if sent:
            out.append(sent)
        buf = []

    for ch in s:
        buf.append(ch)
        if ch in QUOTE_OPEN:
            stack += 1
        elif ch in QUOTE_CLOSE and stack > 0:
            stack -= 1

        # sentence boundary when not inside quotes
        if stack == 0 and ch in "。！？；\n":
            flush()

    if buf:
        flush()
    return [x.strip() for x in out if x.strip()]


def extract_all_quotes(sentence: str) -> List[str]:
    """
    Extract ALL quoted segments in a sentence (non-greedy).
    Returns list of contents without the quote marks.
    """
    quotes = re.findall(r"[“『「](.*?)[”』」]", sentence)
    return [q.strip() for q in quotes if q.strip()]


def remove_quotes(sentence: str) -> str:
    """Remove quoted parts entirely (for narration remainder)."""
    return re.sub(r"[“『「].*?[”』」]", "", sentence).strip()


def normalize_content(s: str) -> str:
    s = s.strip()
    s = BRACKET_RE.sub("", s)  # drop stage directions in brackets
    # drop leading/trailing quotes if still present
    s = s.strip("“”『』「」").strip()
    # remove redundant whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def guess_speaker(raw_sentence: str) -> Optional[str]:
    """
    Guess speaker name from patterns around speech verbs.
    Return a name string (Han 1-6 chars) or None.
    """
    m1 = SPEAKER_BEFORE_RE.search(raw_sentence)
    if m1:
        return m1.group("spk")

    m2 = SPEAKER_AFTER_RE.search(raw_sentence)
    if m2:
        return m2.group("spk")

    return None


def emotion_heuristic(text: str) -> Tuple[str, float]:
    """
    Return (emotion, confidence) using weighted keyword scoring + punctuation signals.
    """
    t = text
    scores: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for pat, emo, w in EMO_SCORES:
        hits = len(pat.findall(t))
        if hits:
            scores[emo] += w * hits

    # punctuation boosts
    if re.search(r"[！!]{2,}", t):
        scores["excited"] += 1.2
        scores["angry"] += 0.6
    elif re.search(r"[！!]", t):
        scores["excited"] += 0.6

    if re.search(r"[？?]{2,}", t):
        scores["surprise"] += 0.8
    elif re.search(r"[？?]", t):
        scores["surprise"] += 0.4

    # pick best
    best = max(scores.items(), key=lambda x: x[1])
    emo, sc = best[0], best[1]
    if sc <= 0.01:
        return "neutral", 0.2
    # map some ambiguous cases
    conf = min(0.95, 0.35 + sc / 4.0)
    return emo, conf


def narration_heuristic(text: str, has_quote: bool) -> bool:
    if has_quote:
        return False
    # if sentence looks like scene/action description, treat as narration
    return bool(NARRATION_HINT_RE.search(text))


def gender_from_roster(speaker: Optional[str], roster: Optional[Dict[str, str]]) -> str:
    if not speaker or not roster:
        return "unknown"
    g = roster.get(speaker)
    if g in ("male", "female"):
        return g
    return "unknown"


@dataclass
class LabelItem:
    id: int
    raw: str
    content: str
    has_quote: bool
    speaker_candidate: Optional[str]
    heuristics: Dict[str, Any]


SYSTEM_PROMPT_V3 = """
You are a strict Chinese text labeler for multi-speaker TTS.
You will receive a JSON ARRAY of objects, each has:
- id (int)
- raw (original sentence)
- content (spoken text candidate, already cleaned)
- has_quote (bool)
- speaker_candidate (string|null)
- heuristics: { emo_hint, emo_conf, narration_hint, gender_hint_from_roster }

Return a JSON ARRAY with EXACTLY the same length and order as input.
Each output item MUST contain ONLY:
{
  "id": <int>,
  "gender": "male|female|unknown",
  "is_narration": true|false,
  "emotion": "neutral|happy|sad|angry|fear|surprise|disgust|calm|excited",
  "content": "<spoken text only>"
}

Rules:
- If has_quote=true, it is dialogue by default => is_narration=false unless content is clearly stage direction.
- Prefer gender_hint_from_roster if provided (male/female). If unknown and speaker_candidate is null, use unknown.
- emotion: follow emo_hint unless raw clearly contradicts; keep it simple.
- Do NOT add extra keys, no markdown, no commentary.
""".strip()


# ---- LLM call placeholder (you will replace with DeepSeek call) ----
def call_llm_label(items: List[LabelItem]) -> List[Dict[str, Any]]:
    """
    Replace this with your DeepSeek call.
    Must return a list of dicts aligned to input ids.
    """
    raise NotImplementedError("Implement call_llm_label() using DeepSeek API.")


def build_items(text: str, roster: Optional[Dict[str, str]] = None) -> List[LabelItem]:
    sentences = split_sentences_quote_aware(text)
    items: List[LabelItem] = []
    idx = 0

    for sent in sentences:
        quotes = extract_all_quotes(sent)
        spk = guess_speaker(sent)

        if quotes:
            # each quote becomes one dialogue item
            for q in quotes:
                content = normalize_content(q)
                if not content:
                    continue
                emo, conf = emotion_heuristic(content)
                items.append(LabelItem(
                    id=idx,
                    raw=sent,
                    content=content,
                    has_quote=True,
                    speaker_candidate=spk,
                    heuristics={
                        "emo_hint": emo,
                        "emo_conf": conf,
                        "narration_hint": False,
                        "gender_hint_from_roster": gender_from_roster(spk, roster),
                    }
                ))
                idx += 1

            # also keep narration remainder if meaningful
            rem = normalize_content(remove_quotes(sent))
            if rem and narration_heuristic(rem, has_quote=False):
                emo, conf = emotion_heuristic(rem)
                items.append(LabelItem(
                    id=idx,
                    raw=sent,
                    content=rem,
                    has_quote=False,
                    speaker_candidate=None,
                    heuristics={
                        "emo_hint": emo,
                        "emo_conf": conf,
                        "narration_hint": True,
                        "gender_hint_from_roster": "unknown",
                    }
                ))
                idx += 1
        else:
            # no quotes: likely narration or indirect speech
            content = normalize_content(sent)
            if not content:
                continue
            narr = narration_heuristic(content, has_quote=False)
            emo, conf = emotion_heuristic(content)
            items.append(LabelItem(
                id=idx,
                raw=sent,
                content=content,
                has_quote=False,
                speaker_candidate=None,
                heuristics={
                    "emo_hint": emo,
                    "emo_conf": conf,
                    "narration_hint": narr,
                    "gender_hint_from_roster": "unknown",
                }
            ))
            idx += 1

    return items


def sanitize_llm_output(arr: List[Dict[str, Any]], items: List[LabelItem]) -> List[Dict[str, Any]]:
    """
    Enforce schema and fix obvious issues.
    """
    by_id = {it.id: it for it in items}
    out: List[Dict[str, Any]] = []

    for obj in arr:
        _id = int(obj.get("id", -1))
        base = by_id.get(_id)
        if base is None:
            continue

        gender = obj.get("gender", "unknown")
        if gender not in GENDERS:
            gender = "unknown"

        is_narr = bool(obj.get("is_narration", False))
        emo = obj.get("emotion", "neutral")
        if emo not in EMOTIONS:
            emo = "neutral"

        content = normalize_content(str(obj.get("content", base.content)))
        if not content:
            continue

        # hard rule: quoted unit => dialogue unless very clearly narration
        if base.has_quote:
            is_narr = False

        out.append({
            "id": _id,
            "gender": gender,
            "is_narration": is_narr,
            "emotion": emo,
            "content": content
        })

    # reorder by id
    out.sort(key=lambda x: x["id"])
    return out


def fallback_label(items: List[LabelItem]) -> List[Dict[str, Any]]:
    """
    Heuristic-only labeling when LLM fails.
    """
    out = []
    for it in items:
        gender_hint = it.heuristics.get("gender_hint_from_roster", "unknown")
        emo = it.heuristics.get("emo_hint", "neutral")
        narr = bool(it.heuristics.get("narration_hint", False)) and not it.has_quote
        out.append({
            "id": it.id,
            "gender": gender_hint if gender_hint in GENDERS else "unknown",
            "is_narration": narr,
            "emotion": emo if emo in EMOTIONS else "neutral",
            "content": it.content
        })
    return out


def label_text(
    text: str,
    roster: Optional[Dict[str, str]] = None,
    use_llm: bool = True,
    out_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool entry: label text into units for TTS.

    Returns:
      {
        "ok": true/false,
        "items": [...],
        "out_jsonl": "... or null",
        "stats": {...}
      }
    """
    t0 = time.time()
    items = build_items(text, roster=roster)

    if not items:
        return {"ok": False, "error": "No valid units extracted.", "items": []}

    labeled: List[Dict[str, Any]] = []
    if use_llm:
        try:
            # You will implement this with DeepSeek. Should return list aligned by id.
            arr = call_llm_label(items)
            labeled = sanitize_llm_output(arr, items)
        except Exception as e:
            labeled = fallback_label(items)
            return {
                "ok": True,
                "items": labeled,
                "out_jsonl": None,
                "stats": {"units": len(labeled), "mode": "fallback", "error": str(e), "elapsed_sec": round(time.time()-t0, 3)}
            }
    else:
        labeled = fallback_label(items)

    out_path = None
    if out_jsonl:
        out_path = str(Path(out_jsonl).expanduser().resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in labeled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "items": labeled,
        "out_jsonl": out_path,
        "stats": {"units": len(labeled), "mode": "llm" if use_llm else "heuristic", "elapsed_sec": round(time.time()-t0, 3)}
    }


def label_text_from_file(
    file_path: str,
    encoding: str = "utf-8",
    roster: Optional[Dict[str, str]] = None,
    use_llm: bool = True,
    out_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return {"ok": False, "error": f"file not found: {p}", "items": []}

    try:
        raw = p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        raw = p.read_text(encoding="gbk", errors="ignore")

    return label_text(raw, roster=roster, use_llm=use_llm, out_jsonl=out_jsonl)
