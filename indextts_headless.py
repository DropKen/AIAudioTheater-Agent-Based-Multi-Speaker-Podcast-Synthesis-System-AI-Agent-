#!/usr/bin/env python3
# indextts_headless.py
import os, sys, time, argparse, json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "indextts"))

from indextts.infer_v2 import IndexTTS2  # 你的项目里已有

def build_tts(model_dir, fp16=False, deepspeed=False, cuda_kernel=False, cfg_name="config.yaml"):
    tts = IndexTTS2(
        model_dir=model_dir,
        cfg_path=os.path.join(model_dir, cfg_name),
        use_fp16=fp16,
        use_deepspeed=deepspeed,
        use_cuda_kernel=cuda_kernel,
    )
    return tts

def run_once(tts, args):
    # 处理情感模式
    emo_mode = args.emo_mode  # speaker/ref/vector/text
    emo_audio = None
    emo_vec = None
    use_emo_text = False
    emo_text = None

    if emo_mode == "speaker":
        emo_audio = None
        emo_vec = None
        use_emo_text = False
    elif emo_mode == "ref":
        emo_audio = args.emo_ref
    elif emo_mode == "vector":
        # 8维向量，和 WebUI 一致
        vec = [args.vec1, args.vec2, args.vec3, args.vec4, args.vec5, args.vec6, args.vec7, args.vec8]
        # WebUI 里有 normalize_emo_vec，可选：
        if hasattr(tts, "normalize_emo_vec"):
            emo_vec = tts.normalize_emo_vec(vec, apply_bias=True)
        else:
            emo_vec = vec
    elif emo_mode == "text":
        use_emo_text = True
        emo_text = args.emo_text or None
    else:
        raise ValueError("emo_mode must be one of: speaker/ref/vector/text")

    out_path = args.out or f"outputs/spk_{int(time.time())}.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    gen_kwargs = dict(
        do_sample=bool(args.do_sample),
        top_p=float(args.top_p),
        top_k=int(args.top_k) if args.top_k > 0 else None,
        temperature=float(args.temperature),
        length_penalty=float(args.length_penalty),
        num_beams=int(args.num_beams),
        repetition_penalty=float(args.repetition_penalty),
        max_mel_tokens=int(args.max_mel_tokens),
    )

    wav_path = tts.infer(
        spk_audio_prompt=args.prompt_audio,   # 可为空；与 WebUI 一致
        text=args.text,
        output_path=out_path,
        emo_audio_prompt=emo_audio,
        emo_alpha=float(args.emo_weight),
        emo_vector=emo_vec,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(args.emo_random),
        verbose=bool(args.verbose),
        max_text_tokens_per_segment=int(args.max_text_tokens_per_segment),
        **gen_kwargs
    )
    print(wav_path)

def main():
    p = argparse.ArgumentParser("IndexTTS-2 Headless CLI")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--cfg_name", default="config.yaml")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--deepspeed", action="store_true")
    p.add_argument("--cuda_kernel", action="store_true")

    # 输入与输出
    p.add_argument("--text", help="文本内容", default=None)
    p.add_argument("--text_file", help="从文件读取文本（优先）", default=None)
    p.add_argument("--prompt_audio", help="音色参考音频", default=None)
    p.add_argument("--out", help="输出 wav 路径", default=None)
    p.add_argument("--verbose", action="store_true")

    # 分句与采样参数（与 WebUI 对齐）
    p.add_argument("--max_text_tokens_per_segment", type=int, default=120)
    p.add_argument("--do_sample", type=int, default=1)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--length_penalty", type=float, default=0.0)
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--repetition_penalty", type=float, default=10.0)
    p.add_argument("--max_mel_tokens", type=int, default=1500)

    # 情感控制
    p.add_argument("--emo_mode", choices=["speaker","ref","vector","text"], default="speaker")
    p.add_argument("--emo_ref", default=None)
    p.add_argument("--emo_weight", type=float, default=0.65)
    p.add_argument("--emo_random", type=int, default=0)
    p.add_argument("--emo_text", default="")

    # 8维向量
    for i in range(1,9):
        p.add_argument(f"--vec{i}", type=float, default=0.0)

    # 批处理（可选）：输入一个 JSON/JSONL 文件，数组或逐行对象
    p.add_argument("--batch_json", help="批量输入文件（JSON 数组或 JSONL，每项需含 text / out 以及可选的情感字段）", default=None)

    args = p.parse_args()

    tts = build_tts(args.model_dir, fp16=args.fp16, deepspeed=args.deepspeed, cuda_kernel=args.cuda_kernel)

    if args.batch_json:
        path = Path(args.batch_json)
        if path.suffix.lower() in [".json"]:
            items = json.loads(Path(path).read_text(encoding="utf-8"))
        else:
            # JSONL
            items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

        for it in items:
            # 允许每条里覆盖 emo 参数
            ns = argparse.Namespace(**vars(args))
            ns.text = it["text"]
            ns.out = it.get("out")
            ns.prompt_audio = it.get("prompt_audio", args.prompt_audio)
            ns.emo_mode = it.get("emo_mode", args.emo_mode)
            ns.emo_ref = it.get("emo_ref", args.emo_ref)
            ns.emo_weight = float(it.get("emo_weight", args.emo_weight))
            ns.emo_random = int(it.get("emo_random", args.emo_random))
            ns.emo_text = it.get("emo_text", args.emo_text)
            for i in range(1,9):
                setattr(ns, f"vec{i}", float(it.get(f"vec{i}", getattr(args, f"vec{i}"))))
            run_once(tts, ns)
    else:
        if args.text_file and not args.text:
            args.text = Path(args.text_file).read_text(encoding="utf-8")
        assert args.text, "必须通过 --text 或 --text_file 提供文本"
        run_once(tts, args)

if __name__ == "__main__":
    main()
