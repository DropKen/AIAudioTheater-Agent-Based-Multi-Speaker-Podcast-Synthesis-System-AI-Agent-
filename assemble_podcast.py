#!/usr/bin/env python3
# assemble_podcast.py
import os, sys, json, glob, math, argparse
from pathlib import Path
from typing import List, Optional
from pydub import AudioSegment
from pydub.effects import normalize as pd_normalize

def load_wav(path: str, target_sr: Optional[int]) -> AudioSegment:
    seg = AudioSegment.from_file(path)
    if target_sr and seg.frame_rate != target_sr:
        seg = seg.set_frame_rate(target_sr)
    # 转单声道更稳（可选）
    if seg.channels != 1:
        seg = seg.set_channels(1)
    return seg

def loudness_normalize(seg: AudioSegment, peak_db: float = -1.0) -> AudioSegment:
    """简单峰值归一到 peak_db（dBFS）。"""
    change = peak_db - seg.max_dBFS
    return seg.apply_gain(change)

def read_paths_from_jsonl(batch_jsonl: str) -> List[str]:
    paths = []
    for line in Path(batch_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        p = obj.get("out")
        if not p:
            raise ValueError("每行需要包含 out 字段（生成的分段音频路径）")
        paths.append(p)
    return paths

def read_paths_from_dir(seg_dir: str, pattern: str = "*.wav") -> List[str]:
    files = sorted(glob.glob(str(Path(seg_dir) / pattern)))
    return files

def add_bgm_mixdown(
    speech: AudioSegment,
    bgm_path: str,
    mix_gain_db: float = -18.0,
    fade_ms: int = 1500
) -> AudioSegment:
    """把 BGM 叠加到整段 speech 上：背景整体降 18dB，首尾淡入淡出。"""
    bgm = AudioSegment.from_file(bgm_path)
    if bgm.channels != speech.channels:
        bgm = bgm.set_channels(speech.channels)
    if bgm.frame_rate != speech.frame_rate:
        bgm = bgm.set_frame_rate(speech.frame_rate)

    # 循环铺满
    if len(bgm) < len(speech):
        times = math.ceil(len(speech) / len(bgm))
        bgm = sum([bgm] * times)

    bgm = bgm[:len(speech)]
    bgm = bgm + mix_gain_db  # 降音量
    if fade_ms > 0:
        bgm = bgm.fade_in(fade_ms).fade_out(fade_ms)
    return speech.overlay(bgm)

def assemble(
    inputs: List[str],
    out_path: str,
    gap_ms: int = 200,
    target_sr: Optional[int] = 24000,
    peak_db: float = -1.0,
    bgm_path: Optional[str] = None,
    bgm_gain_db: float = -18.0,
    bgm_fade_ms: int = 1500,
    export_format: Optional[str] = None
):
    assert inputs, "没有可拼接的段落"
    # 确定导出格式
    if export_format is None:
        export_format = Path(out_path).suffix.lstrip(".").lower() or "wav"

    # 读取与拼接
    final = AudioSegment.silent(duration=0, frame_rate=target_sr or 24000).set_channels(1)
    gap = AudioSegment.silent(duration=max(0, gap_ms), frame_rate=target_sr or 24000).set_channels(1)

    for p in inputs:
        if not Path(p).exists():
            raise FileNotFoundError(f"找不到分段音频：{p}")
        seg = load_wav(p, target_sr)
        # 每段做个轻量峰值归一，避免某段爆音或太小
        seg = loudness_normalize(seg, peak_db=peak_db)
        final += seg + gap

    # 去掉最后一个多余的 gap
    if gap_ms > 0 and len(final) >= gap_ms:
        final = final[:-gap_ms]

    # 给整体再来一次轻量归一
    final = loudness_normalize(final, peak_db=peak_db)

    # 叠加 BGM（可选）
    if bgm_path:
        final = add_bgm_mixdown(final, bgm_path, mix_gain_db=bgm_gain_db, fade_ms=bgm_fade_ms)

    # 导出
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if export_format == "mp3":
        final.export(out_path, format="mp3", bitrate="192k")
    else:
        final.export(out_path, format=export_format)
    print(f"[OK] Saved -> {out_path}")

def main():
    ap = argparse.ArgumentParser("Assemble TTS segments into a single episode")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--batch_jsonl", help="使用 tts_batch.jsonl（按 out 字段顺序）")
    grp.add_argument("--seg_dir", help="从目录读取 *.wav（按文件名排序）")

    ap.add_argument("--out", required=True, help="输出文件，如 outputs/episode.wav or .mp3")
    ap.add_argument("--gap_ms", type=int, default=200, help="段落间停顿毫秒")
    ap.add_argument("--sr", type=int, default=24000, help="目标采样率")
    ap.add_argument("--peak_db", type=float, default=-1.0, help="峰值归一目标 dBFS（-1dB 推荐）")
    ap.add_argument("--bgm", default=None, help="可选：背景音乐文件路径")
    ap.add_argument("--bgm_gain_db", type=float, default=-18.0, help="BGM 整体衰减（dB）")
    ap.add_argument("--bgm_fade_ms", type=int, default=1500, help="BGM 首尾淡入淡出毫秒")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 可执行文件路径，例如 C:\\Users\\28799\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0-full_build\\bin\\ffmpeg.exe")
    args = ap.parse_args()

    if args.batch_jsonl:
        inputs = read_paths_from_jsonl(args.batch_jsonl)
    else:
        inputs = read_paths_from_dir(args.seg_dir, "*.wav")

    if args.ffmpeg:
        AudioSegment.converter = args.ffmpeg
        ffprobe = args.ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
        if os.path.exists(ffprobe):
            AudioSegment.ffprobe = ffprobe

    assemble(
        inputs=inputs,
        out_path=args.out,
        gap_ms=args.gap_ms,
        target_sr=args.sr,
        peak_db=args.peak_db,
        bgm_path=args.bgm,
        bgm_gain_db=args.bgm_gain_db,
        bgm_fade_ms=args.bgm_fade_ms,
        export_format=None
    )

if __name__ == "__main__":
    main()
