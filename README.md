这是一个**聊天式 AI Agent**，把“用户输入的文本/本地 TXT 小说章节”自动转换为**分段语音**并最终合成为**整集音频（MP3/WAV）**。  
Agent 通过大模型理解指令并调用本地工具链（IndexTTS2 + 音频处理脚本）完成端到端流程。

部署方式主要部署index tts model即可，额外添加的库为open ai, 并且需要下载fmmpeg

使用方式：环境部署完后运行agent_run.py 并配置deepseek api即可正常聊天沟通

**核心能力：**
1. **聊天短文本 → 单条音频**
   - 输入：例如「帮我把这段话做成音频：我生气啦！」
   - 输出：单条 `wav`（如 `outputs/single_xxx.wav`）
   - 特点：即时、适合 demo/测试

2. **本地 TXT 小说章节 → labels.jsonl → segments → episode.mp3**
   - 输入：一个本地 `.txt` 路径（章节/短篇）
   - 处理流程：
     - `label_text_from_file`：分句 + 标注（性别 / 是否旁白 / 情绪 / content）
     - `make_tts_batch_jsonl`：生成批处理任务文件 `tts_batch.jsonl`
     - `run_tts_batch`：批量生成 `segments/seg_0000.wav ...`
     - `assemble_episode`：合成 `episode.mp3`（可选 BGM、停顿、响度归一、采样率统一）
   - 特点：可控、可断点续跑、适合批量生产

3. **每个 TXT 自动创建独立项目目录（防止混合）**
   - 目录示例：
     ```
     outputs/projects/chapter1_YYYYMMDD_HHMMSS/
       labels.jsonl
       tts_batch.jsonl
       segments/seg_0000.wav ...
       episode.mp3
     outputs/projects/chapter2_YYYYMMDD_HHMMSS/
       ...
     ```
   - 解决问题：多个 txt 章节不会把 segments 混在一起，合成不会串内容

4. **工具化（Tools-first）设计，支持 LLM Tool Calling**
   - 文本工具：`label_text` / `label_text_from_file`
   - 批处理工具：`make_tts_batch_jsonl` / `run_tts_batch`
   - TTS 工具：`tts_generate`
   - 合成工具：`assemble_episode`
   - 调度入口：`agent_run.py`（聊天式交互）

5. **鲁棒性与兜底**
   - LLM 余额不足/不可用时：可走本地工具执行（不影响核心流水线）
   - segments 可跳过已存在文件（便于断点续跑）