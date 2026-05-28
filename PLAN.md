# 本地自动化 Sydney 数据工厂 + Unsloth 训练 Notebook 导出

## 已实现范围

- 本地 FastAPI 单页 Web UI。
- 自动蓝图规划，无需手动输入主题。
- Teacher 默认使用 OpenAI Responses API 数据生成，可切回 Chat Completions。
- Judge 默认使用 OpenAI Responses API 自动审核；不配置时复用 Teacher。
- 启发式审核兜底。
- 相关性、风格、多轮质量、模板腔、安全、训练价值评分。
- 近重复检测与自动丢弃。
- SQLite 工作区持久化。
- accepted / needs_review / rejected 人工复核。
- ShareGPT / ChatML JSONL 导出。
- Unsloth 训练 Notebook 导出，支持：
  - `unsloth/Qwen3.6-27B`
  - `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
  - `qlora` / `lora` / `fft`

## 文件结构

```text
sydney_NEWBING/
├─ app.py
├─ data_generator.py
├─ auto_reviewer.py
├─ deduper.py
├─ notebook_exporter.py
├─ prompts.py
├─ requirements-dataset.txt
├─ README.md
├─ PLAN.md
├─ .env.example
└─ data/
   ├─ workspace.sqlite
   ├─ raw/
   ├─ reviewed/
   ├─ rejected/
   └─ exports/
```

## 默认策略

- 数据生成：Teacher 负责；默认协议 `responses`，对应 `/v1/responses`。
- 自动审核：Judge 优先，失败则启发式审核；Judge 默认协议同为 `responses`。
- 兼容模式：将 `*_API_PROTOCOL` 改成 `chat_completions` 可调用 `/v1/chat/completions`。
- 重复检测：字符 n-gram + SequenceMatcher。
- 导出：只导出 accepted；可选包含 needs_review。
- Notebook：默认 QLoRA。
