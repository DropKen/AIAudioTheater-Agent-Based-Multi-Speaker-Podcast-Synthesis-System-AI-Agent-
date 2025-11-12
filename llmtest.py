# llm_segmenter_bigmodel_v2.py
import os, sys, time, json, regex as re, requests
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

API_KEY  = "59c0ce54af214efc9568cbb178944362.R1mDIbETla0gn19a"
API_BASE = "https://open.bigmodel.cn/api/paas/v4"
MODEL    = "glm-4.6"
TIMEOUT  = 120
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

EMOTIONS = ["neutral","happy","sad","angry","fear","surprise","disgust","calm","excited"]
GENDERS  = ["male","female","unknown"]

# === 关键词启发式 ===
EMO_HINTS = [
    (re.compile(r"(怒|生气|火|吵|提高音量|青筋|拍桌|咆哮)"), "angry"),
    (re.compile(r"(哭|泪|哽咽|委屈|难过)"), "sad"),
    (re.compile(r"(惊|愣|突然|骤然|一下子)"), "surprise"),
    (re.compile(r"(笑|开心|高兴|愉快|噗嗤笑)"), "happy"),
    (re.compile(r"(叹气|平静|温和|镇定|缓缓)"), "calm"),
]
NARR_HINT = re.compile(r"(门铃|火锅咕嘟|安静|场景|他|她|看着|走去|站着|抱着|快步|突然|霎时|只听见)")
FEMALE_PRON = re.compile(r"(她|她的)")
MALE_PRON   = re.compile(r"(他|他的)")

SYSTEM_PROMPT = """You are a careful Chinese text labeler for multi-speaker podcast TTS.
You will receive a list of JSON objects. Each object has:
{
  "raw": "<original sentence>",
  "quote_content": "<text extracted from quotes if any, else empty>",
  "heuristics": {
    "is_quote": true|false,
    "emo_hint": "<angry|sad|happy|surprise|calm|none>",
    "gender_hint": "<male|female|unknown>",
    "narration_hint": true|false
  }
}
Your task: For EACH object, output EXACTLY one item with ONLY these fields:
{
  "gender": "male|female|unknown",
  "is_narration": true|false,
  "content": "<spoken text only, no speaker name, no quotes>",
  "emotion": "neutral|happy|sad|angry|fear|surprise|disgust|calm|excited"
}

Rules:
- If is_quote=true, treat it as DIALOGUE by default unless clearly stage direction. Use quote_content as content after trimming.
- If is_quote=false but the sentence is purely scene description, set is_narration=true and gender="unknown".
- Use heuristics as hard evidence when reasonable (emo_hint/gender_hint/narration_hint). You may correct them if obviously wrong from the raw text.
- Remove speaker names or brackets, keep only the spoken words.
- Keep essential punctuation for prosody. Do not add commentary.
- Return a JSON ARRAY ONLY. No markdown or extra keys.
"""

FEWSHOT = [
    {
      "raw":"“我吃饱了。”",
      "quote_content":"我吃饱了。",
      "heuristics":{"is_quote":True,"emo_hint":"neutral","gender_hint":"female","narration_hint":False}
    },
    {
      "raw":"她声音生硬，眼圈微红，起身就要离开。",
      "quote_content":"",
      "heuristics":{"is_quote":False,"emo_hint":"sad","gender_hint":"female","narration_hint":True}
    },
    {
      "raw":"“每次都是加班！”",
      "quote_content":"每次都是加班！",
      "heuristics":{"is_quote":True,"emo_hint":"angry","gender_hint":"male","narration_hint":False}
    }
]

def split_paragraphs(text:str)->List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

# quote 感知分句：尽量把引号内当作一句
QUOTE_PAIRS = [("“","”"),("『','』"),("「","」")]
def quote_aware_split(text:str)->List[str]:
    # 先按强终止符切，再合并跨引号段
    rough = [s for s in re.split(r"(?<=[。！？；…])\s*|\n+", text) if s and s.strip()]
    out=[]
    buf=""
    inside=False
    for s in rough:
        buf = buf + ("" if not buf else "") + s.strip()
        # 统计引号平衡
        opens = len(re.findall(r"[“『「]", buf))
        closes= len(re.findall(r"[”』」]", buf))
        if opens==closes:
            out.append(buf); buf=""
    if buf: out.append(buf)
    return out

def extract_quote(s:str)->str:
    # 提取第一对中文引号里的文本
    m = re.search(r"[“『「](.*?)[”』」]", s)
    return (m.group(1).strip() if m else "")

def heuristics_for(s:str)->Dict[str,Any]:
    is_quote = bool(extract_quote(s))
    # 情绪线索
    emo="none"
    for pat,label in EMO_HINTS:
        if pat.search(s):
            emo=label; break
    # 性别线索
    g="unknown"
    if FEMALE_PRON.search(s): g="female"
    elif MALE_PRON.search(s): g="male"
    narr = bool(NARR_HINT.search(s)) and not is_quote
    return {"is_quote":is_quote, "emo_hint":emo, "gender_hint":g, "narration_hint":narr}

def chunk(items:List[Dict[str,Any]], n:int=10)->List[List[Dict[str,Any]]]:
    res=[]; buf=[]
    for it in items:
        buf.append(it)
        if len(buf)>=n: res.append(buf); buf=[]
    if buf: res.append(buf)
    return res

def call_bigmodel(objs:List[Dict[str,Any]])->List[Dict[str,Any]]:
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": "下面是三个示例（你只需学习风格和输出格式）：" },
        {"role":"user","content": json.dumps(FEWSHOT, ensure_ascii=False)},
        {"role":"user","content": "现在请标注以下对象列表（逐条输出数组）：" },
        {"role":"user","content": json.dumps(objs, ensure_ascii=False)}
    ]
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    start = content.find('['); end = content.rfind(']')
    if start==-1 or end==-1 or end<start:
        raise ValueError("Model did not return a JSON array.")
    arr = json.loads(content[start:end+1])
    cleaned=[]
    for item in arr:
        cleaned.append({
            "gender": item.get("gender","unknown") if item.get("gender","unknown") in GENDERS else "unknown",
            "is_narration": bool(item.get("is_narration", False)),
            "content": str(item.get("content","")).strip(),
            "emotion": item.get("emotion","neutral") if item.get("emotion","neutral") in EMOTIONS else "neutral",
        })
    return [x for x in cleaned if x["content"]]

def apply_roster(lines:List[Dict[str,Any]], roster:Dict[str,str]) -> List[Dict[str,Any]]:
    # 如果 content 或 raw 附近提到某姓名，则用 roster 兜底性别
    names = list(roster.keys())
    name_pat = re.compile("|".join(map(re.escape,names))) if names else None
    out=[]
    for l in lines:
        if l["gender"]=="unknown" and name_pat and name_pat.search(l["content"]):
            nm = name_pat.search(l["content"]).group(0)
            l["gender"] = "female" if roster.get(nm)=="female" else "male"
        out.append(l)
    return out

def process_text(raw:str, batch_items:int=10, roster:Dict[str,str]=None):
    sents=[]
    for p in [p.strip() for p in re.split(r"\n\s*\n+", raw.strip()) if p.strip()]:
        sents += quote_aware_split(p)
    objs=[]
    for s in sents:
        objs.append({
            "raw": s,
            "quote_content": extract_quote(s),
            "heuristics": heuristics_for(s)
        })
    results=[]
    for b in tqdm(chunk(objs, n=batch_items), desc="LLM labeling (v2)"):
        for _ in range(3):
            try:
                out = call_bigmodel(b)
                results += out; break
            except Exception as e:
                time.sleep(1.3)
        else:
            # 三次失败兜底：启发式生成
            for o in b:
                isq = o["heuristics"]["is_quote"]
                content = o["quote_content"] if isq and o["quote_content"] else o["raw"]
                emo = o["heuristics"]["emo_hint"] if o["heuristics"]["emo_hint"]!="none" else "neutral"
                gender = o["heuristics"]["gender_hint"]
                narr = (not isq) or o["heuristics"]["narration_hint"]
                results.append({"gender":gender,"is_narration":narr,"content":content,"emotion":emo})
    if roster:
        results = apply_roster(results, roster)
    # 简单规则：引号内默认不是旁白
    for r in results:
        if re.search(r"^[“『「].*[”』」]$", r["content"]):
            r["content"] = re.sub(r'^[“『「]|[”』」]$', "", r["content"]).strip()
            r["is_narration"] = False
    return results

def main():
    import argparse
    ap = argparse.ArgumentParser("BigModel segment+label v2 (quote-aware + heuristics)")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--batch_items", type=int, default=10)
    ap.add_argument("--roster", default=None, help="可选：角色-性别映射 JSON，如 {\"李萌\":\"female\",\"陈浩\":\"male\"}")
    args = ap.parse_args()

    raw = Path(args.infile).read_text(encoding="utf-8")
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8")) if args.roster else None
    rows = process_text(raw, batch_items=args.batch_items, roster=roster)

    outp = Path(args.outfile); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)); f.write("\n")
    print(f"Done. {len(rows)} lines -> {outp}")

if __name__ == "__main__":
    main()
