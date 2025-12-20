#!/usr/bin/env python3
# tools.py
"""
Unified tools for your audiobook agent:
1) Text labeling (gender/emotion/narration) from text or txt file
2) IndexTTS2 synthesis from text or txt file
3) Assemble wav segments into episode (wav/mp3) via pydub (+ optional ffmpeg path)

All tool outputs are JSON-serializable dicts, suitable for agent tool calling.
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave
import glob
import math
import threading
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import regex as re  # keep your dependency

DEFAULT_MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
DEFAULT_PROMPT_AUDIO = os.getenv("INDEXTTS_PROMPT_AUDIO", "examples/voice_01.wav")

# =========================================================
# Common helpers
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

def _resolve_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return str(Path(p).expanduser().resolve())

def _ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def _now_ts() -> int:
    return int(time.time())

def _read_wav_info(wav_path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
        duration = (nframes / sr) if sr else None
        info.update(
            sample_rate=sr,
            n_frames=nframes,
            channels=nch,
            sample_width=sampwidth,
            duration_sec=duration,
        )
    except Exception as e:
        info.update(wav_info_error=str(e))
    return info

def _parse_roster(roster: Optional[Union[Dict[str, str], str]]) -> Optional[Dict[str, str]]:
    """
    roster can be:
      - dict: {"李萌":"female","陈浩":"male"}
      - json string: same
      - file path to json: e.g. "./roster.json"
    """
    if roster is None:
        return None
    if isinstance(roster, dict):
        return roster
    s = str(roster).strip()
    if not s:
        return None
    p = Path(s)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    # otherwise treat as json string
    return json.loads(s)


# =========================================================
# (A) Text labeling tools (heuristics-first, improved)
# =========================================================
EMOTIONS = ["neutral","happy","sad","angry","fear","surprise","disgust","calm","excited"]
GENDERS  = ["male","female","unknown"]

QUOTE_OPEN = "“『「"
QUOTE_CLOSE = "”』」"

# Speech verbs for speaker candidate extraction
SAY_VERBS = [
    "说","问","答","道","喊","叫","吼","嚷","嘀咕","嘟囔","喃喃",
    "低声说","轻声说","笑道","哭道","骂道","冷笑","提醒","解释","劝","命令","宣布","咆哮","尖叫"
]
SAY_VERB_RE = re.compile("|".join(map(re.escape, sorted(SAY_VERBS, key=len, reverse=True))))

# Speaker patterns: “张三说：” / “……”，张三说
SPEAKER_BEFORE_RE = re.compile(
    rf"(?P<spk>[\p{{Han}}]{{1,6}})(?:[^\p{{Han}}]{{0,3}})?(?:{SAY_VERB_RE.pattern})(?:[：:，, ]|$)"
)
SPEAKER_AFTER_RE = re.compile(
    rf"(?:[，,。！!？?]\s*)?(?P<spk>[\p{{Han}}]{{1,6}})(?:[^\p{{Han}}]{{0,3}})?(?:{SAY_VERB_RE.pattern})"
)

# Emotion scoring: weighted multi-hit + punctuation signals
EMO_SCORES: List[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"(怒|生气|火冒|拍桌|咆哮|吼|嚷|骂|瞪)"), "angry", 2.0),
    (re.compile(r"(哭|泪|哽咽|抽泣|委屈|难过|心酸|悲伤)"), "sad", 2.0),
    (re.compile(r"(害怕|恐惧|发抖|颤|惊恐|尖叫|躲)"), "fear", 2.0),
    (re.compile(r"(震惊|愣|突然|骤然|一下子|诧异)"), "surprise", 1.5),
    (re.compile(r"(笑|开心|高兴|愉快|兴奋|雀跃)"), "happy", 1.5),
    (re.compile(r"(平静|温和|镇定|缓缓|轻轻|柔声)"), "calm", 1.2),
    (re.compile(r"(恶心|厌恶|嫌弃|作呕)"), "disgust", 1.5),
    (re.compile(r"(激动|亢奋|热血|振奋)"), "excited", 1.2),
]

# Narration hints: avoid using “他/她” as narration hint (too generic)
NARRATION_HINT_RE = re.compile(
    r"(四周|夜色|阳光|风|雨|雷|空气|街道|屋里|门外|脚步声|沉默|安静|钟声|灯光|墙|桌|椅|走|站|坐|靠|转身|抬手|握|抱|躲|快步)"
)

BRACKET_RE = re.compile(r"[\(\（\[\【].*?[\)\）\]\】]")


def _split_sentences_quote_aware(text: str) -> List[str]:
    """
    Split by 。！？； and newlines but do NOT cut inside quotes.
    """
    s = text.strip()
    if not s:
        return []
    out: List[str] = []
    buf: List[str] = []
    stack = 0

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

        if stack == 0 and ch in "。！？；\n":
            flush()

    if buf:
        flush()

    return [x.strip() for x in out if x.strip()]


def _extract_all_quotes(sentence: str) -> List[str]:
    quotes = re.findall(r"[“『「](.*?)[”』」]", sentence)
    return [q.strip() for q in quotes if q.strip()]

def _remove_quotes(sentence: str) -> str:
    return re.sub(r"[“『「].*?[”』」]", "", sentence).strip()

def _normalize_content(s: str) -> str:
    s = s.strip()
    s = BRACKET_RE.sub("", s)
    s = s.strip("“”『』「」").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _guess_speaker(raw_sentence: str) -> Optional[str]:
    m1 = SPEAKER_BEFORE_RE.search(raw_sentence)
    if m1:
        return m1.group("spk")
    m2 = SPEAKER_AFTER_RE.search(raw_sentence)
    if m2:
        return m2.group("spk")
    return None

def _gender_from_roster(speaker: Optional[str], roster: Optional[Dict[str, str]]) -> str:
    if not speaker or not roster:
        return "unknown"
    g = roster.get(speaker)
    return g if g in ("male", "female") else "unknown"

def _emotion_heuristic(text: str) -> Tuple[str, float]:
    scores: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for pat, emo, w in EMO_SCORES:
        hits = len(pat.findall(text))
        if hits:
            scores[emo] += w * hits

    # punctuation
    if re.search(r"[！!]{2,}", text):
        scores["excited"] += 1.2
        scores["angry"] += 0.6
    elif re.search(r"[！!]", text):
        scores["excited"] += 0.6

    if re.search(r"[？?]{2,}", text):
        scores["surprise"] += 0.8
    elif re.search(r"[？?]", text):
        scores["surprise"] += 0.4

    emo, sc = max(scores.items(), key=lambda x: x[1])
    if sc <= 0.01:
        return "neutral", 0.2
    conf = min(0.95, 0.35 + sc / 4.0)
    return emo, conf

def _narration_heuristic(text: str, has_quote: bool) -> bool:
    if has_quote:
        return False
    return bool(NARRATION_HINT_RE.search(text))


def label_text(
    text: str,
    roster: Optional[Union[Dict[str, str], str]] = None,
    use_llm: bool = False,
    out_jsonl: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tool: label_text
    - Heuristics-first labeling for TTS units.
    - If use_llm=True, you can later plug DeepSeek labeling; currently stays heuristics-only.

    Returns:
      { ok, items:[{id,gender,is_narration,emotion,content,speaker_candidate}], stats, out_jsonl }
    """
    t0 = time.time()
    roster_dict = _parse_roster(roster)

    sents = _split_sentences_quote_aware(text)
    items: List[Dict[str, Any]] = []
    uid = 0

    for sent in sents:
        spk = _guess_speaker(sent)
        quotes = _extract_all_quotes(sent)

        if quotes:
            # each quote => dialogue unit
            for q in quotes:
                content = _normalize_content(q)
                if not content:
                    continue
                emo, conf = _emotion_heuristic(content)
                gender = _gender_from_roster(spk, roster_dict)
                items.append({
                    "id": uid,
                    "gender": gender,
                    "is_narration": False,
                    "emotion": emo,
                    "content": content,
                    "speaker_candidate": spk,
                    "emo_conf": conf
                })
                uid += 1

            # optional narration remainder if meaningful
            rem = _normalize_content(_remove_quotes(sent))
            if rem and _narration_heuristic(rem, has_quote=False):
                emo, conf = _emotion_heuristic(rem)
                items.append({
                    "id": uid,
                    "gender": "unknown",
                    "is_narration": True,
                    "emotion": emo,
                    "content": rem,
                    "speaker_candidate": None,
                    "emo_conf": conf
                })
                uid += 1
        else:
            content = _normalize_content(sent)
            if not content:
                continue
            narr = _narration_heuristic(content, has_quote=False)
            emo, conf = _emotion_heuristic(content)
            items.append({
                "id": uid,
                "gender": "unknown",
                "is_narration": narr,
                "emotion": emo,
                "content": content,
                "speaker_candidate": None,
                "emo_conf": conf
            })
            uid += 1

    if not items:
        return {"ok": False, "error": "No valid units extracted.", "items": []}

    # (Optional) place for LLM refinement, left disabled by default
    if use_llm:
        # You can implement DeepSeek labeling here later.
        # For now we keep heuristics result to avoid breaking your pipeline.
        pass

    outp = None
    if out_jsonl:
        outp = str(Path(out_jsonl).expanduser().resolve())
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "items": items,
        "out_jsonl": outp,
        "stats": {"units": len(items), "mode": "heuristic" if not use_llm else "heuristic+llm(todo)", "elapsed_sec": round(time.time()-t0, 3)}
    }


def label_text_from_file(
    file_path: str,
    encoding: str = "utf-8",
    roster: Optional[Union[Dict[str, str], str]] = None,
    use_llm: bool = False,
    out_jsonl: Optional[str] = None
) -> Dict[str, Any]:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return {"ok": False, "error": f"file not found: {p}", "items": []}
    try:
        raw = p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        raw = p.read_text(encoding="gbk", errors="ignore")
    return label_text(raw, roster=roster, use_llm=use_llm, out_jsonl=out_jsonl)


# =========================================================
# (B) IndexTTS2 tools (direct cached OR subprocess)
# =========================================================
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "indextts"))

try:
    from indextts.infer_v2 import IndexTTS2  # type: ignore
    _HAS_DIRECT_TTS = True
except Exception:
    IndexTTS2 = None  # type: ignore
    _HAS_DIRECT_TTS = False


@dataclass
class TTSRequest:
    model_dir: str = DEFAULT_MODEL_DIR
    cfg_name: str = "config.yaml"
    fp16: bool = False
    deepspeed: bool = False
    cuda_kernel: bool = False

    text: str = ""
    prompt_audio: Optional[str] = DEFAULT_PROMPT_AUDIO
    out_path: Optional[str] = None

    emo_mode: str = "speaker"     # speaker/ref/vector/text
    emo_ref: Optional[str] = None
    emo_weight: float = 0.65
    emo_random: int = 0
    emo_text: str = ""
    emo_vector: Optional[List[float]] = None  # len=8

    max_text_tokens_per_segment: int = 120
    do_sample: int = 1
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500

    verbose: bool = False
    overwrite: bool = True

    backend: str = "direct"  # direct/subprocess


_TTS_CACHE: Dict[Tuple[str, str, bool, bool, bool], Any] = {}
_TTS_LOCK = threading.Lock()

def _get_tts(model_dir: str, cfg_name: str, fp16: bool, deepspeed: bool, cuda_kernel: bool):
    if not _HAS_DIRECT_TTS:
        raise RuntimeError("Direct mode not available (IndexTTS2 import failed). Use backend='subprocess'.")
    key = (str(Path(model_dir).resolve()), cfg_name, fp16, deepspeed, cuda_kernel)
    with _TTS_LOCK:
        if key not in _TTS_CACHE:
            _TTS_CACHE[key] = IndexTTS2(
                model_dir=model_dir,
                cfg_path=os.path.join(model_dir, cfg_name),
                use_fp16=fp16,
                use_deepspeed=deepspeed,
                use_cuda_kernel=cuda_kernel,
            )
        return _TTS_CACHE[key]

def _infer_direct(req: TTSRequest) -> str:
    tts = _get_tts(req.model_dir, req.cfg_name, req.fp16, req.deepspeed, req.cuda_kernel)

    emo_mode = req.emo_mode
    emo_audio = None
    emo_vec = None
    use_emo_text = False
    emo_text = None

    if emo_mode == "speaker":
        pass
    elif emo_mode == "ref":
        emo_audio = req.emo_ref
    elif emo_mode == "vector":
        vec = req.emo_vector or [0.0] * 8
        if len(vec) != 8:
            raise ValueError("emo_vector must have length 8 when emo_mode='vector'")
        if hasattr(tts, "normalize_emo_vec"):
            emo_vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            emo_vec = vec
    elif emo_mode == "text":
        use_emo_text = True
        emo_text = req.emo_text or None
    else:
        raise ValueError("emo_mode must be one of: speaker/ref/vector/text")

    out_path = req.out_path or f"outputs/spk_{_now_ts()}.wav"
    out_path = _resolve_path(out_path) or out_path
    _ensure_parent_dir(out_path)

    if (not req.overwrite) and Path(out_path).exists():
        return out_path

    gen_kwargs = dict(
        do_sample=bool(req.do_sample),
        top_p=float(req.top_p),
        top_k=int(req.top_k) if req.top_k > 0 else None,
        temperature=float(req.temperature),
        length_penalty=float(req.length_penalty),
        num_beams=int(req.num_beams),
        repetition_penalty=float(req.repetition_penalty),
        max_mel_tokens=int(req.max_mel_tokens),
    )

    wav_path = tts.infer(
        spk_audio_prompt=_resolve_path(req.prompt_audio),
        text=req.text,
        output_path=out_path,
        emo_audio_prompt=_resolve_path(emo_audio),
        emo_alpha=float(req.emo_weight),
        emo_vector=emo_vec,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(req.emo_random),
        verbose=bool(req.verbose),
        max_text_tokens_per_segment=int(req.max_text_tokens_per_segment),
        **gen_kwargs
    )
    return str(Path(wav_path).resolve())

def _infer_subprocess(req: TTSRequest) -> str:
    py = sys.executable or "python"
    script = PROJECT_ROOT / "indextts_headless.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find indextts_headless.py at: {script}")

    out_path = req.out_path or f"outputs/spk_{_now_ts()}.wav"
    out_path = _resolve_path(out_path) or out_path
    _ensure_parent_dir(out_path)

    if (not req.overwrite) and Path(out_path).exists():
        return out_path

    cmd = [
        py, str(script),
        "--model_dir", req.model_dir,
        "--cfg_name", req.cfg_name,
        "--text", req.text,
        "--out", out_path,
        "--emo_mode", req.emo_mode,
        "--emo_weight", str(req.emo_weight),
        "--emo_random", str(req.emo_random),
        "--emo_text", req.emo_text or "",
        "--max_text_tokens_per_segment", str(req.max_text_tokens_per_segment),
        "--do_sample", str(req.do_sample),
        "--top_p", str(req.top_p),
        "--top_k", str(req.top_k),
        "--temperature", str(req.temperature),
        "--length_penalty", str(req.length_penalty),
        "--num_beams", str(req.num_beams),
        "--repetition_penalty", str(req.repetition_penalty),
        "--max_mel_tokens", str(req.max_mel_tokens),
    ]

    if req.fp16: cmd.append("--fp16")
    if req.deepspeed: cmd.append("--deepspeed")
    if req.cuda_kernel: cmd.append("--cuda_kernel")
    if req.verbose: cmd.append("--verbose")

    if req.prompt_audio:
        cmd += ["--prompt_audio", req.prompt_audio]
    if req.emo_mode == "ref" and req.emo_ref:
        cmd += ["--emo_ref", req.emo_ref]
    if req.emo_mode == "vector":
        vec = req.emo_vector or [0.0] * 8
        if len(vec) != 8:
            raise ValueError("emo_vector must have length 8 when emo_mode='vector'")
        for i, v in enumerate(vec, start=1):
            cmd += [f"--vec{i}", str(v)]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            "TTS subprocess failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
    # prefer printed wav path if exists
    printed = (proc.stdout or "").strip().splitlines()
    if printed:
        last = printed[-1].strip()
        if last.lower().endswith(".wav") and Path(last).exists():
            return str(Path(last).resolve())
    return out_path


def tts_generate(**kwargs) -> Dict[str, Any]:
    kwargs.setdefault("model_dir", DEFAULT_MODEL_DIR)
    kwargs.setdefault("prompt_audio", DEFAULT_PROMPT_AUDIO)
    req = TTSRequest(**kwargs)
    req.model_dir = _resolve_path(req.model_dir) or req.model_dir

    if not Path(req.model_dir).exists():
        return {"ok": False, "error": f"model_dir not found: {req.model_dir}", "request": asdict(req)}
    if not req.text or not req.text.strip():
        return {"ok": False, "error": "text is empty", "request": asdict(req)}

    if req.backend not in ("direct", "subprocess"):
        return {"ok": False, "error": "backend must be 'direct' or 'subprocess'", "request": asdict(req)}
    if req.backend == "direct" and not _HAS_DIRECT_TTS:
        req.backend = "subprocess"

    try:
        wav_path = _infer_direct(req) if req.backend == "direct" else _infer_subprocess(req)
        meta = _read_wav_info(wav_path)
        return {"ok": True, "wav_path": wav_path, "meta": meta, "request": asdict(req)}
    except Exception as e:
        return {"ok": False, "error": str(e), "request": asdict(req)}


def tts_generate_from_file(
    file_path: str,
    encoding: str = "utf-8",
    max_chars: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return {"ok": False, "error": f"file not found: {p}", "file_path": str(p)}
    try:
        text = p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        text = p.read_text(encoding="gbk", errors="ignore")
    if max_chars and max_chars > 0:
        text = text[:max_chars]
    return tts_generate(text=text, **kwargs)


# =========================================================
# (C) Assemble episode tool (pydub)
# =========================================================
def assemble_episode(
    out_path: str,
    seg_dir: Optional[str] = None,
    batch_jsonl: Optional[str] = None,
    pattern: str = "*.wav",
    gap_ms: int = 200,
    sr: int = 24000,
    peak_db: float = -1.0,
    bgm: Optional[str] = None,
    bgm_gain_db: float = -18.0,
    bgm_fade_ms: int = 1500,
    ffmpeg: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tool: assemble_episode
    - Provide either seg_dir (read *.wav sorted) or batch_jsonl (read out field order)
    - Output can be .wav or .mp3 (mp3 requires ffmpeg to be available to pydub)
    """
    if (seg_dir is None) == (batch_jsonl is None):
        return {"ok": False, "error": "Provide exactly one of seg_dir or batch_jsonl."}

    out_path = _resolve_path(out_path) or out_path
    _ensure_parent_dir(out_path)

    # lazy import so tools.py can load even if pydub isn't installed yet
    try:
        from pydub import AudioSegment
    except Exception as e:
        return {"ok": False, "error": f"pydub import failed: {e}. Try: pip install pydub"}

    # setup ffmpeg if provided (especially for mp3)
    if ffmpeg:
        ffmpeg_path = _resolve_path(ffmpeg)
        if not ffmpeg_path or not Path(ffmpeg_path).exists():
            return {"ok": False, "error": f"ffmpeg not found: {ffmpeg}"}
        AudioSegment.converter = ffmpeg_path
        ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
        if os.path.exists(ffprobe):
            AudioSegment.ffprobe = ffprobe

    def load_wav(path: str, target_sr: Optional[int]) -> "AudioSegment":
        seg = AudioSegment.from_file(path)
        if target_sr and seg.frame_rate != target_sr:
            seg = seg.set_frame_rate(target_sr)
        if seg.channels != 1:
            seg = seg.set_channels(1)
        return seg

    def loudness_normalize(seg: "AudioSegment", peak_db_: float = -1.0) -> "AudioSegment":
        change = peak_db_ - seg.max_dBFS
        return seg.apply_gain(change)

    def read_paths_from_jsonl(pth: str) -> List[str]:
        p = Path(pth).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"batch_jsonl not found: {p}")
        paths = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out = obj.get("out") or obj.get("wav_path")
            if not out:
                raise ValueError("Each jsonl line must contain 'out' or 'wav_path'.")
            paths.append(out)
        return paths

    def read_paths_from_dir(d: str, pat: str) -> List[str]:
        dd = Path(d).expanduser().resolve()
        if not dd.exists():
            raise FileNotFoundError(f"seg_dir not found: {dd}")
        files = sorted(glob.glob(str(dd / pat)))
        return files

    def add_bgm_mixdown(speech: "AudioSegment", bgm_path: str, mix_gain_db: float, fade_ms: int) -> "AudioSegment":
        bgm_seg = AudioSegment.from_file(bgm_path)
        if bgm_seg.channels != speech.channels:
            bgm_seg = bgm_seg.set_channels(speech.channels)
        if bgm_seg.frame_rate != speech.frame_rate:
            bgm_seg = bgm_seg.set_frame_rate(speech.frame_rate)
        if len(bgm_seg) < len(speech):
            times = math.ceil(len(speech) / len(bgm_seg))
            bgm_seg = sum([bgm_seg] * times)
        bgm_seg = bgm_seg[:len(speech)]
        bgm_seg = bgm_seg + mix_gain_db
        if fade_ms > 0:
            bgm_seg = bgm_seg.fade_in(fade_ms).fade_out(fade_ms)
        return speech.overlay(bgm_seg)

    try:
        inputs = read_paths_from_jsonl(batch_jsonl) if batch_jsonl else read_paths_from_dir(seg_dir, pattern)
        if not inputs:
            return {"ok": False, "error": "No input segments found."}

        final = AudioSegment.silent(duration=0, frame_rate=sr).set_channels(1)
        gap = AudioSegment.silent(duration=max(0, gap_ms), frame_rate=sr).set_channels(1)

        for p in inputs:
            pp = Path(p)
            if not pp.exists():
                return {"ok": False, "error": f"segment not found: {p}"}
            seg = load_wav(str(pp), sr)
            seg = loudness_normalize(seg, peak_db_=peak_db)
            final += seg + gap

        if gap_ms > 0 and len(final) >= gap_ms:
            final = final[:-gap_ms]

        final = loudness_normalize(final, peak_db_=peak_db)

        if bgm:
            bgm_path = _resolve_path(bgm) or bgm
            if not Path(bgm_path).exists():
                return {"ok": False, "error": f"bgm not found: {bgm_path}"}
            final = add_bgm_mixdown(final, bgm_path, mix_gain_db=bgm_gain_db, fade_ms=bgm_fade_ms)

        fmt = Path(out_path).suffix.lstrip(".").lower() or "wav"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "mp3":
            # requires ffmpeg available to pydub
            final.export(out_path, format="mp3", bitrate="192k")
        else:
            final.export(out_path, format=fmt)

        return {
            "ok": True,
            "out_path": out_path,
            "segments": len(inputs),
            "duration_sec": round(len(final) / 1000.0, 3),
            "format": fmt
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================================================
# Tool specs + dispatcher (for your agent)
# =========================================================
TOOL_SPECS = [
    {
        "type": "function",
        "name": "label_text",
        "description": "Split and label Chinese text into TTS units with emotion/narration hints.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "roster": {"type": ["object", "string", "null"], "description": "dict/json/or json file path mapping name->gender"},
                "use_llm": {"type": "boolean", "default": False},
                "out_jsonl": {"type": ["string", "null"], "description": "optional jsonl output path"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "label_text_from_file",
        "description": "Label a local .txt file into TTS units with emotion/narration hints.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "roster": {"type": ["object", "string", "null"]},
                "use_llm": {"type": "boolean", "default": False},
                "out_jsonl": {"type": ["string", "null"]},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "tts_generate",
        "description": "Generate a wav audio file from text using local IndexTTS2.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string"},
                "cfg_name": {"type": "string", "default": "config.yaml"},
                "fp16": {"type": "boolean", "default": False},
                "deepspeed": {"type": "boolean", "default": False},
                "cuda_kernel": {"type": "boolean", "default": False},

                "text": {"type": "string"},
                "prompt_audio": {"type": ["string", "null"]},
                "out_path": {"type": ["string", "null"]},

                "emo_mode": {"type": "string", "enum": ["speaker","ref","vector","text"], "default": "speaker"},
                "emo_ref": {"type": ["string", "null"]},
                "emo_weight": {"type": "number", "default": 0.65},
                "emo_random": {"type": "integer", "default": 0},
                "emo_text": {"type": "string", "default": ""},
                "emo_vector": {"type": ["array", "null"], "items": {"type": "number"}},

                "max_text_tokens_per_segment": {"type": "integer", "default": 120},
                "do_sample": {"type": "integer", "default": 1},
                "top_p": {"type": "number", "default": 0.8},
                "top_k": {"type": "integer", "default": 30},
                "temperature": {"type": "number", "default": 0.8},
                "length_penalty": {"type": "number", "default": 0.0},
                "num_beams": {"type": "integer", "default": 3},
                "repetition_penalty": {"type": "number", "default": 10.0},
                "max_mel_tokens": {"type": "integer", "default": 1500},

                "verbose": {"type": "boolean", "default": False},
                "overwrite": {"type": "boolean", "default": True},
                "backend": {"type": "string", "enum": ["direct","subprocess"], "default": "direct"},
            },
            "required": ["model_dir", "text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "tts_generate_from_file",
        "description": "Generate a wav audio file from a local txt file path using IndexTTS2.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_chars": {"type": ["integer","null"]},

                "model_dir": {"type": "string"},
                "cfg_name": {"type": "string", "default": "config.yaml"},
                "prompt_audio": {"type": ["string","null"]},
                "out_path": {"type": ["string","null"]},
                "emo_mode": {"type": "string", "enum": ["speaker","ref","vector","text"], "default": "speaker"},
                "emo_ref": {"type": ["string","null"]},
                "emo_weight": {"type": "number", "default": 0.65},
                "backend": {"type": "string", "enum": ["direct","subprocess"], "default": "direct"},
            },
            "required": ["file_path", "model_dir"],
            "additionalProperties": True,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "assemble_episode",
        "description": "Assemble wav segments into one episode wav/mp3 (pydub). Provide seg_dir OR batch_jsonl.",
        "parameters": {
            "type": "object",
            "properties": {
                "out_path": {"type": "string"},
                "seg_dir": {"type": ["string","null"]},
                "batch_jsonl": {"type": ["string","null"]},
                "pattern": {"type": "string", "default": "*.wav"},
                "gap_ms": {"type": "integer", "default": 200},
                "sr": {"type": "integer", "default": 24000},
                "peak_db": {"type": "number", "default": -1.0},
                "bgm": {"type": ["string","null"]},
                "bgm_gain_db": {"type": "number", "default": -18.0},
                "bgm_fade_ms": {"type": "integer", "default": 1500},
                "ffmpeg": {"type": ["string","null"]},
            },
            "required": ["out_path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

def dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "label_text":
        return label_text(**arguments)
    if name == "label_text_from_file":
        return label_text_from_file(**arguments)
    if name == "tts_generate":
        return tts_generate(**arguments)
    if name == "tts_generate_from_file":
        return tts_generate_from_file(**arguments)
    if name == "assemble_episode":
        return assemble_episode(**arguments)
    if name == "prepare_project_dir":
        return prepare_project_dir(**arguments)
    if name == "make_tts_batch_jsonl":
        return make_tts_batch_jsonl(**arguments)
    if name == "run_tts_batch":
        return run_tts_batch(**arguments)
    return {"ok": False, "error": f"Unknown tool: {name}", "name": name}



# =========================
# (D) Project + Batch tools
# =========================
import re as _re

DEFAULT_PROMPT_MAP = {
    ("female", False): "prompts/emo_hate.wav",
    ("male",   False): "prompts/emo_sad.wav",
    ("unknown", True): "prompts/voice_01.wav",   # narration
    ("female", True):  "prompts/voice_02.wav",
    ("male",   True):  "prompts/voice_03.wav",
    ("unknown", False): "prompts/voice_05.wav",
}

def _safe_slug(name: str) -> str:
    # Windows-friendly folder name
    name = name.strip()
    name = _re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name)
    name = _re.sub(r"\s+", "_", name)
    return name[:80] if len(name) > 80 else name

def prepare_project_dir(
    input_txt: str,
    output_root: str = "outputs/projects",
    project_name: str | None = None,
    unique: bool = True
) -> dict:
    """
    Create a dedicated folder for this txt so multiple books/chapters won't mix.
    Returns {ok, project_dir, segments_dir, labels_jsonl, tts_batch_jsonl, episode_path}
    """
    p = Path(input_txt).expanduser().resolve()
    if not p.exists():
        return {"ok": False, "error": f"input_txt not found: {p}"}

    base = project_name or p.stem
    base = _safe_slug(base) or "project"
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if unique:
        # add timestamp to avoid collisions
        ts = time.strftime("%Y%m%d_%H%M%S")
        proj = root / f"{base}_{ts}"
    else:
        proj = root / base

    seg_dir = proj / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    return {
        "ok": True,
        "project_dir": str(proj),
        "segments_dir": str(seg_dir),
        "labels_jsonl": str(proj / "labels.jsonl"),
        "tts_batch_jsonl": str(proj / "tts_batch.jsonl"),
        "episode_path": str(proj / "episode.mp3"),
        "input_txt": str(p),
    }

def _pick_prompt(gender: str, is_narration: bool, prompt_map: dict) -> str:
    g = gender if gender in ("male", "female", "unknown") else "unknown"
    return prompt_map.get((g, bool(is_narration)), prompt_map.get(("unknown", False), "prompts/guest_neutral.wav"))

def make_tts_batch_jsonl(
    in_labels_jsonl: str,
    out_batch_jsonl: str,
    out_dir: str,
    prompt_map: dict | None = None,
    emo_mode: str = "speaker",
    overwrite: bool = True
) -> dict:
    """
    Convert labeled jsonl (each line has gender/is_narration/content/emotion)
    -> tts_batch.jsonl (each line has text/prompt_audio/out/emo_mode...)
    Each sentence => one segment wav.
    """
    prompt_map = prompt_map or DEFAULT_PROMPT_MAP

    in_path = Path(in_labels_jsonl).expanduser().resolve()
    if not in_path.exists():
        return {"ok": False, "error": f"in_labels_jsonl not found: {in_path}"}

    out_batch = Path(out_batch_jsonl).expanduser().resolve()
    out_batch.parent.mkdir(parents=True, exist_ok=True)

    seg_dir = Path(out_dir).expanduser().resolve()
    seg_dir.mkdir(parents=True, exist_ok=True)

    lines = in_path.read_text(encoding="utf-8").splitlines()
    out_rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        ex = json.loads(line)

        prompt_audio = _pick_prompt(ex.get("gender", "unknown"), ex.get("is_narration", False), prompt_map)

        seg_path = seg_dir / f"seg_{i:04d}.wav"
        if seg_path.exists() and (not overwrite):
            # keep existing path, still include it in batch
            pass

        out_rows.append({
            "text": ex.get("content", ""),
            "prompt_audio": prompt_audio,
            "emo_mode": emo_mode,     # later you can map emotion -> ref/vector here
            "out": str(seg_path),
        })

    with out_batch.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"ok": True, "count": len(out_rows), "out_batch_jsonl": str(out_batch), "segments_dir": str(seg_dir)}

def run_tts_batch(
    batch_jsonl: str,
    model_dir: str,
    cfg_name: str = "config.yaml",
    backend: str = "direct",
    fp16: bool = False,
    deepspeed: bool = False,
    cuda_kernel: bool = False,
    overwrite: bool = True,
    stop_on_error: bool = False,
    verbose: bool = False
) -> dict:
    """
    Read tts_batch.jsonl and synthesize each line into its 'out' wav.
    Uses cached direct IndexTTS2 if possible.
    Returns stats + failed list.
    """
    batch_path = Path(batch_jsonl).expanduser().resolve()
    if not batch_path.exists():
        return {"ok": False, "error": f"batch_jsonl not found: {batch_path}"}

    lines = [ln for ln in batch_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    ok_n = 0
    failed = []

    for idx, ln in enumerate(lines):
        item = json.loads(ln)
        text = (item.get("text") or "").strip()
        out = item.get("out")
        if not text or not out:
            failed.append({"index": idx, "error": "missing text/out", "item": item})
            if stop_on_error:
                break
            continue

        # skip existing if overwrite=False
        if (not overwrite) and Path(out).exists():
            ok_n += 1
            continue

        res = tts_generate(
            model_dir=model_dir,
            cfg_name=cfg_name,
            fp16=fp16,
            deepspeed=deepspeed,
            cuda_kernel=cuda_kernel,
            text=text,
            prompt_audio=item.get("prompt_audio"),
            out_path=out,
            emo_mode=item.get("emo_mode", "speaker"),
            emo_ref=item.get("emo_ref"),
            emo_weight=float(item.get("emo_weight", 0.65)),
            emo_random=int(item.get("emo_random", 0)),
            emo_text=item.get("emo_text", ""),
            emo_vector=item.get("emo_vector"),
            verbose=verbose,
            overwrite=overwrite,
            backend=backend,
        )

        if res.get("ok"):
            ok_n += 1
        else:
            failed.append({"index": idx, "error": res.get("error"), "item": item})
            if stop_on_error:
                break

    return {
        "ok": (ok_n == total),
        "total": total,
        "succeeded": ok_n,
        "failed": failed[:50],  # avoid huge return
        "batch_jsonl": str(batch_path),
    }
