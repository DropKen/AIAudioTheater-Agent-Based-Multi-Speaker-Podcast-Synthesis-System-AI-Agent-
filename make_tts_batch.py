# make_tts_batch_with_prompts.py
import json, sys
from pathlib import Path

# 为不同性别/旁白配置音色参考音频
PROMPT_MAP = {
    ("female", False): "prompts/emo_hate.wav",
    ("male",   False): "prompts/emo_sad.wav",
    ("unknown",True):  "prompts/voice_01.wav",   # 旁白
    ("female", True):  "prompts/voice_02.wav",
    ("male",   True):  "prompts/voice_03.wav",
    ("unknown",False): "prompts/voice_05.wav",
}

def pick_prompt(gender, is_narration):
    return PROMPT_MAP.get((gender, bool(is_narration)), "prompts/guest_neutral.wav")

def main(in_jsonl, out_jsonl, out_dir="outputs/segments"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lines = Path(in_jsonl).read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.strip(): continue
        ex = json.loads(line)
        prompt_audio = pick_prompt(ex.get("gender","unknown"), ex.get("is_narration", False))
        item = {
            "text": ex["content"],
            "prompt_audio": prompt_audio,
            "emo_mode": "speaker",                 # 先用最稳的；你也可按 emotion 改成 vector/ref/text
            "out": str(Path(out_dir)/f"seg_{i:04d}.wav")
        }
        out.append(item)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False)); f.write("\n")
    print(f"Wrote {len(out)} -> {out_jsonl}")

if __name__ == "__main__":
    in_jsonl, out_jsonl = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv)>=4 else "outputs/segments"
    main(in_jsonl, out_jsonl, out_dir)
