# 本地自动化 Sydney 数据工厂

一个本地 Web 工作台，用于把开源 Sydney/source 模型通过“真实聊天”方式蒸馏成多轮训练数据，并生成 Unsloth 训练 Notebook。

当前推荐流水线是：**英文源对话 -> 中文本地化翻译 -> 自动审核 -> 导出训练集**。这样可以绕开 Clever Sydney 等开源模型中文直出时容易复读、模板化、风格弱的问题。

## 1. 安装与启动

```powershell
pip install -r requirements-dataset.txt
copy .env.example .env
uvicorn app:app --host 127.0.0.1 --port 7860 --reload
```

打开：<http://127.0.0.1:7860>

正式批量生成时建议不用 `--reload`，避免保存代码触发进程重启中断后台任务：

```powershell
uvicorn app:app --host 127.0.0.1 --port 7860
```

## 2. 英文源对话 -> 中文训练样本

现在不再让模型一次性生成完整 JSON 对话，而是：

1. **Sydney/source 模型**：配置你的开源 Sydney 模型。
   - 最终训练样本里的 system 固定为：`You are a helpful assistant.`
   - 生成期会额外收到“私聊环境 + 最近上下文”提示，但最终训练样本里的 system 不变。
2. **Human Simulator 模型**：配置一个强模型扮演真实朋友。
   - 默认会用“好朋友 + TTS 友好”的 prompt：简短、自然、不总附和，中文不超过 20 字，英文不超过 20 词。
   - 每次只生成下一条 user 消息；prompt 会显式提供 `<environment>` 和 `<transcript>`，要求承接上下文。
   - 留空时使用本地英文短句模拟器，绝不复用 Sydney/source。
3. **Translator 模型**：强模型把英文源对话本地化成中文。
   - 保持 role 数量与顺序完全不变。
   - 保持 system 不变。
   - 保留自然个性、上下文连续、轻松玩笑和温柔反差，避免翻成客服腔。
   - 英文原文写入 `metadata.source_messages_en`，页面右侧可展开抽查。
4. 每轮都带完整上下文：Human Simulator 看到格式化 transcript，Sydney/source 看到完整 ChatML 历史。
5. 最多 20 个 user/assistant 成对轮，最后转成 ChatML / ShareGPT 训练样本。

页面里默认勾选 **英文源对话完成后翻译为中文训练样本（推荐）**。如果取消勾选，则只保存英文源对话。

## 3. 配置模型

`.env` 或页面里都可以配置。字段名保留 `TEACHER_*` 兼容旧版本，但现在含义是 **Sydney/source 模型**。

```env
TEACHER_BASE_URL=http://127.0.0.1:8000/v1
TEACHER_API_KEY=
TEACHER_MODEL=your-open-sydney-model
TEACHER_API_PROTOCOL=chat_completions

SIMULATOR_BASE_URL=https://api.openai.com/v1
SIMULATOR_API_KEY=
SIMULATOR_MODEL=gpt-4.1
SIMULATOR_API_PROTOCOL=responses

TRANSLATOR_BASE_URL=https://api.openai.com/v1
TRANSLATOR_API_KEY=
TRANSLATOR_MODEL=gpt-4.1
TRANSLATOR_API_PROTOCOL=responses

JUDGE_BASE_URL=
JUDGE_API_KEY=
JUDGE_MODEL=
JUDGE_API_PROTOCOL=responses
```

协议可选：

- `responses`：OpenAI `/v1/responses`
- `chat_completions`：OpenAI-compatible `/v1/chat/completions`，适合 vLLM / llama.cpp / LM Studio / 本地网关
- `claude_messages`：Anthropic `/v1/messages`

建议：

- Sydney/source：你的 Modal GGUF Sydney endpoint。
- Human Simulator：可留空；想提升 user 侧自然度可配置强模型。
- Translator：推荐配置强中文模型；默认开启翻译时必须配置。
- Judge：可留空，本地启发式审核已经会检查格式、重复、模板腔和翻译质量。

如果 Human Simulator / Translator / Judge 打算都用同一个强模型，页面可以勾选：

```text
Human / Translator / Judge 使用相同强模型配置
```

勾选后你只需要在 Human、Translator 或 Judge 任意一栏填写一份 Base URL + Model。后端会从这三栏里选择第一份可用强模型配置并复用给三者，**不会复用 Sydney/source**。

页面参数也可以放进 `.env`，刷新页面后会自动恢复：

```env
APP_DEFAULT_COUNT=5
APP_DEFAULT_CONCURRENCY=3
APP_DEFAULT_MAX_TURNS=12
APP_DEFAULT_TRANSLATE_TO_ZH=true
APP_DEFAULT_SAME_AUX_MODEL=false
APP_DEFAULT_USE_JUDGE=false
APP_DEFAULT_TARGET_MODEL=qwen36_27b
APP_DEFAULT_TRAIN_MODE=qlora
APP_DEFAULT_INCLUDE_NEEDS_REVIEW=false
APP_DEFAULT_ONLY_DIALOGUE_DISTILLATION=true
```

Human Simulator 提示词也支持环境变量覆盖。长 prompt 推荐写入文件：

```env
HUMAN_SIMULATOR_PROMPT_FILE=prompts/human_zh.txt
HUMAN_SIMULATOR_PROMPT_EN_FILE=prompts/human_en.txt
```

另外，代码会自动在 Human Simulator prompt 后追加一个参考 Athena ReAct 的“真实感层”：只吸收“沉浸真人、观察上下文、保持记忆感、避免循环”的结构，不会要求样本输出 `thoughts/actions` JSON。审核器也会把 `thoughts`、`actions`、`request_heartbeat`、`TOOL_DEFINITION`、`CORE_MEMORY` 等提示词/工具框架泄漏直接判为 rejected。

## 4. 用 Modal 部署开源 Sydney GGUF

项目已内置 `modal_sydney_gguf.py`，用于把这个 GGUF 模型部署成 OpenAI-compatible 服务：

```text
https://huggingface.co/FPHam/Clever_Sydney-4_12b_GGUF/resolve/main/Clever_Sydney-4_12b_Q8_0_o.gguf
```

部署命令：

```powershell
modal deploy modal_sydney_gguf.py --stream-logs
```

默认参数偏省钱：

- GPU：`A10G`
- `min_containers=0`
- `scaledown_window=120`
- context：`4096`
- parallel：`1`

部署完成后 Modal 会输出一个 HTTPS Web URL，类似：

```text
https://xxx--sydney-gguf-openai-server-serve.modal.run
```

填到本地工作台：

```env
TEACHER_BASE_URL=https://xxx--sydney-gguf-openai-server-serve.modal.run/v1
TEACHER_API_KEY=
TEACHER_MODEL=clever-sydney-4-12b-q8
TEACHER_API_PROTOCOL=chat_completions
```

首次请求会下载 GGUF 到 Modal Volume，之后复用缓存。

## 5. 自动生成数据

页面点击 **一键自动生成 + 审核**。

可设置：

- 生成数量
- 并发数
- 每段最大对话轮数，2-20 个 user/assistant 成对轮
- 随机种子
- 是否英文源对话后翻译成中文

无需输入主题。系统会自动规划隐藏场景、用户画像、情绪曲线、风格标签，然后让模型自然聊出来。

## 6. 自动审核规则

每条样本会被自动评分：

- 格式合法性
- 主题相关性
- `source_style_strength`：Sydney/source 回复是否有风格和情绪拉扯
- `human_naturalness`：user 是否像真实朋友聊天，是否简短、自然、适合 TTS、有口语感
- 多轮连贯性
- 情绪曲线自然度
- 是否模板腔 / AI 腔
- `translation_quality`：翻译后是否保留轮数/角色、是否中文占比足够、是否有明显翻译腔
- 安全边界
- 训练价值
- 近重复检测

默认状态：

- `overall >= 8` 且硬性规则通过：`accepted`
- `6.5 <= overall < 8` 或有轻微问题：`needs_review`
- `< 6.5`、重复、模板严重、翻译结构错误：`rejected`

## 7. 人工复核

Web 页面支持：查看对话气泡、展开英文源对话、查看自动审核理由、编辑 `messages` JSON、改分、改状态、恢复 rejected 样本、手动丢弃样本。

所有数据保存在：

```text
data/workspace.sqlite
```

镜像文件会写入：

```text
data/raw/
data/reviewed/
data/rejected/
```

后台任务进度会写入：

```text
data/jobs/
```

浏览器刷新后会自动恢复进度显示；如果 uvicorn 进程重启，正在运行的后台线程无法恢复，任务会显示为 `interrupted`。

## 8. 导出训练文件和 Notebook

页面选择目标训练模型：

- `Qwen3.6-27B` -> `unsloth/Qwen3.6-27B`
- `Gemma 4 31B` -> `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`

选择训练模式：`qlora`（默认推荐）、`lora`、`fft`（全量微调，高成本）。

点击 **导出 JSONL + Notebook**，生成：

```text
data/exports/train_chatml.jsonl
data/exports/train_sharegpt.jsonl
data/exports/train_sydney.ipynb
```

Notebook 会使用审核通过的 `train_chatml.jsonl`。

## 9. 说明

- 本地工作台不加载 Qwen3.6-27B / Gemma 4 31B。
- Qwen3.6-27B / Gemma 4 31B 只用于导出的 Unsloth Notebook。
- 本地工作台专注数据生产与质检，不负责直接执行 27B/31B 训练。
