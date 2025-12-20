#!/usr/bin/env python3
# agent_run.py
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI, APIStatusError

import tools


# -----------------------
# Config via env vars
# -----------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Your local defaults
INDEXTTS_MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
FFMPEG_EXE = os.getenv("FFMPEG_EXE", "")  # optional, needed for mp3 on Windows
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "outputs/projects")

# If you want background music automatically
DEFAULT_BGM = os.getenv("BGM_PATH", "")  # optional


def is_txt_path(s: str) -> Optional[str]:
    """If user input looks like a local .txt path, return normalized path; else None."""
    s = s.strip().strip('"').strip("'")
    if not s:
        return None
    p = Path(s).expanduser()
    if p.exists() and p.is_file() and p.suffix.lower() == ".txt":
        return str(p.resolve())
    return None


def run_full_pipeline_from_txt(
    txt_path: str,
    ffmpeg_exe: Optional[str] = None,
    bgm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Offline/local full pipeline:
      txt -> project dir -> label jsonl -> tts_batch jsonl -> segments wav -> assemble mp3
    """
    # 1) Prepare project folder (unique per txt)
    proj = tools.prepare_project_dir(
        input_txt=txt_path,
        output_root=OUTPUT_ROOT,
        project_name=None,
        unique=True,
    )
    if not proj.get("ok"):
        return proj

    labels_jsonl = proj["labels_jsonl"]
    tts_batch_jsonl = proj["tts_batch_jsonl"]
    segments_dir = proj["segments_dir"]
    episode_path = proj["episode_path"]

    # 2) Label text -> labels.jsonl
    lab = tools.label_text_from_file(
        file_path=txt_path,
        use_llm=False,            # heuristics first; later you can set True
        out_jsonl=labels_jsonl,
    )
    if not lab.get("ok"):
        return {"ok": False, "stage": "label", "error": lab.get("error"), "detail": lab}

    # 3) Make tts_batch.jsonl -> each line => one segment wav
    bat = tools.make_tts_batch_jsonl(
        in_labels_jsonl=labels_jsonl,
        out_batch_jsonl=tts_batch_jsonl,
        out_dir=segments_dir,
        emo_mode="speaker",
        overwrite=True,
    )
    if not bat.get("ok"):
        return {"ok": False, "stage": "make_batch", "error": bat.get("error"), "detail": bat}

    # 4) Run batch TTS -> generate segments wav
    run = tools.run_tts_batch(
        batch_jsonl=tts_batch_jsonl,
        model_dir=INDEXTTS_MODEL_DIR,
        backend="direct",
        overwrite=True,
        stop_on_error=False,
        verbose=False,
    )
    if not run.get("ok"):
        # still proceed to assemble if most succeeded? here we stop and return info
        return {"ok": False, "stage": "tts_batch", "error": "some segments failed", "detail": run, "project": proj}

    # 5) Assemble to mp3
    asm = tools.assemble_episode(
        out_path=episode_path,
        seg_dir=segments_dir,
        gap_ms=200,
        sr=24000,
        peak_db=-1.0,
        bgm=(bgm_path or DEFAULT_BGM) or None,
        ffmpeg=(ffmpeg_exe or FFMPEG_EXE) or None,
    )
    if not asm.get("ok"):
        return {"ok": False, "stage": "assemble", "error": asm.get("error"), "detail": asm, "project": proj}

    return {"ok": True, "project": proj, "label": lab.get("stats"), "batch": bat, "tts": run, "assemble": asm}


def tts_single_text_local(text: str) -> Dict[str, Any]:
    """Offline/local: single text -> one wav"""
    out_path = str((Path("outputs") / f"single_{int(time.time())}.wav").resolve())
    return tools.tts_generate(
        model_dir=INDEXTTS_MODEL_DIR,
        prompt_audio="examples/voice_01.wav",
        text=text,
        out_path=out_path,
        backend="direct",
        emo_mode="speaker",
    )


def build_deepseek_tools() -> list[dict]:
    """
    DeepSeek expects tools format:
      [{"type":"function","function":{"name":...,"description":...,"parameters":...}}]
    We'll build a minimal schema for our important tools.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "prepare_project_dir",
                "description": "Create an isolated output folder for this txt input (avoids mixing segments across books).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_txt": {"type": "string"},
                        "output_root": {"type": "string", "default": OUTPUT_ROOT},
                        "project_name": {"type": ["string", "null"]},
                        "unique": {"type": "boolean", "default": True},
                    },
                    "required": ["input_txt"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "label_text_from_file",
                "description": "Read a local .txt and produce labeled units (gender/emotion/narration/content). Can also write labels.jsonl.",
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
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_tts_batch_jsonl",
                "description": "Convert labels.jsonl -> tts_batch.jsonl. Each line becomes one wav segment under segments/.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "in_labels_jsonl": {"type": "string"},
                        "out_batch_jsonl": {"type": "string"},
                        "out_dir": {"type": "string"},
                        "prompt_map": {"type": ["object", "null"]},
                        "emo_mode": {"type": "string", "default": "speaker"},
                        "overwrite": {"type": "boolean", "default": True},
                    },
                    "required": ["in_labels_jsonl", "out_batch_jsonl", "out_dir"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tts_batch",
                "description": "Read tts_batch.jsonl and synthesize each line into its 'out' wav. Uses local IndexTTS2.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "batch_jsonl": {"type": "string"},
                        "model_dir": {"type": "string", "default": INDEXTTS_MODEL_DIR},
                        "cfg_name": {"type": "string", "default": "config.yaml"},
                        "backend": {"type": "string", "enum": ["direct", "subprocess"], "default": "direct"},
                        "overwrite": {"type": "boolean", "default": True},
                        "stop_on_error": {"type": "boolean", "default": False},
                        "verbose": {"type": "boolean", "default": False},
                    },
                    "required": ["batch_jsonl"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "assemble_episode",
                "description": "Assemble wav segments into one episode mp3/wav using pydub. Provide seg_dir OR batch_jsonl.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "out_path": {"type": "string"},
                        "seg_dir": {"type": ["string", "null"]},
                        "batch_jsonl": {"type": ["string", "null"]},
                        "pattern": {"type": "string", "default": "*.wav"},
                        "gap_ms": {"type": "integer", "default": 200},
                        "sr": {"type": "integer", "default": 24000},
                        "peak_db": {"type": "number", "default": -1.0},
                        "bgm": {"type": ["string", "null"]},
                        "bgm_gain_db": {"type": "number", "default": -18.0},
                        "bgm_fade_ms": {"type": "integer", "default": 1500},
                        "ffmpeg": {"type": ["string", "null"]},
                    },
                    "required": ["out_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tts_generate",
                "description": "Generate one wav from a piece of text using local IndexTTS2.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_dir": {"type": "string", "default": INDEXTTS_MODEL_DIR},
                        "text": {"type": "string"},
                        "prompt_audio": {"type": ["string", "null"]},
                        "out_path": {"type": ["string", "null"]},
                        "emo_mode": {"type": "string", "enum": ["speaker", "ref", "vector", "text"], "default": "speaker"},
                        "emo_ref": {"type": ["string", "null"]},
                        "emo_weight": {"type": "number", "default": 0.65},
                        "backend": {"type": "string", "enum": ["direct", "subprocess"], "default": "direct"},
                    },
                    "required": ["text"],
                    "additionalProperties": True,
                },
            },
        },
    ]


def dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # Patch defaults so model doesn't have to repeat
    if name == "run_tts_batch":
        args.setdefault("model_dir", INDEXTTS_MODEL_DIR)
    if name == "assemble_episode":
        args.setdefault("ffmpeg", FFMPEG_EXE or None)
    return tools.dispatch_tool_call(name, args)


def main():
    # If you don't have DeepSeek balance, agent still works via offline local pipeline.
    client = None
    if DEEPSEEK_API_KEY:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    system = f"""
你是“有声书制作助手”。用户通过聊天给你：
- 一段中文文本：你要生成单条语音 wav
- 或一个本地 .txt 文件路径：你要把它做成“分句 segments + 最终 episode.mp3”

默认规则：
- .txt 文件：必须创建独立 project 目录，避免不同 txt 的 segments 混在一起
- 流程：prepare_project_dir -> label_text_from_file(out_jsonl=labels.jsonl) -> make_tts_batch_jsonl -> run_tts_batch -> assemble_episode
- TTS model_dir 默认：{INDEXTTS_MODEL_DIR}
- 输出根目录：{OUTPUT_ROOT}
- 如果需要 mp3，Windows 建议提供 ffmpeg 路径（可用环境变量 FFMPEG_EXE）
""".strip()

    messages = [{"role": "system", "content": system}]
    tools_list = build_deepseek_tools()

    print("=== AudioBook Agent (DeepSeek + Local Tools) ===")
    print("Input: a piece of text or a .txt file path")
    print("Order：/reset  /exit\n")

    while True:
        user = input("You> ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [{"role": "system", "content": system}]
            print("已重置对话。\n")
            continue

        # Fast path: if user gives a txt path, run full local pipeline immediately (no need to spend LLM tokens)
        txt = is_txt_path(user)
        if txt:
            print("AI> Received txt path, starting local processing: annotation -> segment TTS -> synthesis....\n")
            result = run_full_pipeline_from_txt(txt, ffmpeg_exe=FFMPEG_EXE or None, bgm_path=DEFAULT_BGM or None)
            if result.get("ok"):
                ep = result["assemble"]["out_path"]
                dur = result["assemble"].get("duration_sec")
                segs = result["assemble"].get("segments")
                print(f"AI> ✅ 完成！\n- 输出：{ep}\n- 段落数：{segs}\n- 时长：{dur}s\n- 项目目录：{result['project']['project_dir']}\n")
            else:
                print(f"AI> ❌ 失败：{result.get('stage','unknown')} - {result.get('error')}\n详情：{result.get('detail')}\n")
            continue

        # Otherwise: use DeepSeek tool calling if available; else local single-tts fallback
        if client is None:
            # offline fallback: single text -> wav
            print("AI> DeepSeek Not configured (or don't want to use it). I'll start with local single-sentence synthesis.\n")
            res = tts_single_text_local(user.replace("Turn this text into an audio clip:", "").strip())
            if res.get("ok"):
                print(f"AI> ✅ success：{res['wav_path']}（{res.get('meta',{}).get('duration_sec','?')}s）\n")
            else:
                print(f"AI> ❌ fail：{res.get('error')}\n")
            continue

        messages.append({"role": "user", "content": user})

        while True:
            try:
                resp = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=tools_list,
                    tool_choice="auto",
                    stream=False,
                )
            except APIStatusError as e:
                # DeepSeek余额不足: 402
                if getattr(e, "status_code", None) == 402:
                    print("AI> DeepSeek insufficient balance (402). I'll first use local single-sentence synthesis.\n")
                    res = tts_single_text_local(user.replace("Turn this text into an audio clip:", "").strip())
                    if res.get("ok"):
                        print(f"AI> ✅ success：{res['wav_path']}（{res.get('meta',{}).get('duration_sec','?')}s）\n")
                    else:
                        print(f"AI> ❌ fail：{res.get('error')}\n")
                    break
                raise

            msg = resp.choices[0].message

            # tool calls
            if getattr(msg, "tool_calls", None):
                messages.append(msg)
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments or "{}")
                    result = dispatch_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                # continue loop until no more tool calls
                continue

            # normal assistant message
            messages.append({"role": "assistant", "content": msg.content or ""})
            print(f"AI> {msg.content}\n")
            break


if __name__ == "__main__":
    main()
