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
   - 现在默认使用更贴近传统聊天器的调用方式：生成期不给 Sydney/source 额外注入 system prompt，只传完整 `user/assistant` 历史。
   - 如果你的模型需要旧版生成期上下文提示，可设置 `SOURCE_PROMPT_MODE=context_system`。
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
5. 最多 20 个 user/assistant 成对轮；达到最小轮数后，Human 模型会根据完整上下文判断是否已经自然结束，最后转成 ChatML / ShareGPT 训练样本。

页面里默认勾选 **英文源对话完成后翻译为中文训练样本（推荐）**。如果取消勾选，则只保存英文源对话。

## 3. 配置模型

`.env` 或页面里都可以配置。字段名保留 `TEACHER_*` 兼容旧版本，但现在含义是 **Sydney/source 模型**。

```env
TEACHER_BASE_URL=http://127.0.0.1:8000/v1
TEACHER_API_KEY=
TEACHER_MODEL=your-open-sydney-model
TEACHER_API_PROTOCOL=legacy_chat_completions

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
- `legacy_chat_completions`：同样调用 `/v1/chat/completions`，但用于 Sydney/source 时按传统聊天器格式构造上下文，不注入 system prompt，推荐给 Clever Sydney / GGUF 聊天模型
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

# Sydney/source 生成格式。默认 legacy_chat：传统聊天器格式，只传 user/assistant 历史。
SOURCE_PROMPT_MODE=legacy_chat
# 默认不给 Sydney/source 发送 stop，避免传统聊天器输出被硬截断。
SOURCE_USE_DEFAULT_STOPS=false
# 默认不按固定长度截断模型输出，只清理明显角色/ChatML 泄漏。
GENERATION_PRESERVE_LENGTH=true
# 达到该轮数后，Human 模型开始判断是否自然结束。
GENERATION_MIN_TURNS=6
# Sydney/source 采样与反复读控制。实时趋势审核会在此基础上动态提高惩罚。
SOURCE_TEMPERATURE=0.78
SOURCE_TOP_P=0.92
SOURCE_FREQUENCY_PENALTY=0.35
SOURCE_PRESENCE_PENALTY=0.25
SOURCE_REPEAT_PENALTY=1.12
SOURCE_MAX_TOKENS=512
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

## 5. AMD/ROCm 平台一键部署 Sydney GGUF

如果你使用的是 AMD GPU 云环境，例如 `ubuntu22.04-rocm7.x-py312-torch...`，可以直接用内置脚本自动配置 `llama.cpp + ROCm/HIP`，把 Sydney GGUF 跑成 OpenAI-compatible API。

把项目里的脚本上传到云环境后执行：

```bash
bash scripts/setup_sydney_rocm.sh --start
```

如果平台不能直接暴露 `8000` 端口，使用临时 Cloudflare Tunnel：

```bash
bash scripts/setup_sydney_rocm.sh --start --tunnel
```

如果卡在“下载 cloudflared”，可以用国内代理源重跑，脚本会短超时后自动尝试备用源：

```bash
CLOUDFLARED_URL=https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64   bash scripts/setup_sydney_rocm.sh --start --tunnel
```

如果你已经在本地下载好了 `cloudflared-linux-amd64`，可以上传到云环境的 `/mnt/cloudflared`，然后执行：

```bash
chmod +x /mnt/cloudflared
CLOUDFLARED_LOCAL_PATH=/mnt/cloudflared bash scripts/setup_sydney_rocm.sh --start --tunnel
```

脚本会优先复制 `/mnt/cloudflared`，不再从 GitHub 下载。

如果仍失败，Sydney API 本身已经在 `8000` 端口启动，可以不用 `--tunnel`，改用平台自带的端口转发/公开端口功能。

脚本会自动完成：

- 安装 git/cmake/ninja/curl 等基础依赖
- 克隆并编译 ROCm/HIP 版 llama.cpp
- 自动检测 `AMDGPU_TARGETS`，也可手动覆盖
- 下载 `Clever_Sydney-4_12b_Q8_0_o.gguf`
- 生成启动/停止/状态脚本
- 输出本地工作台 `.env` 配置片段

常用参数：

```bash
# 推荐：192GB 显存可先用 8 个 llama.cpp slot；本地工作台并发也建议先 8-16
CTX_SIZE=4096 PARALLEL=8 bash scripts/setup_sydney_rocm.sh --start

# 更激进：如果显存和吞吐都扛得住，可尝试 16；再高通常会被单模型吞吐/排队收益限制
CTX_SIZE=4096 PARALLEL=16 bash scripts/setup_sydney_rocm.sh --start

# 手动指定 AMD 架构，适合自动检测失败时
AMDGPU_TARGETS=gfx942 bash scripts/setup_sydney_rocm.sh --start

# 更换工作目录到持久盘
WORKDIR=/mnt/data/sydney_rocm bash scripts/setup_sydney_rocm.sh --start --tunnel
```


### 高并发建议

`llama.cpp --parallel` 表示 server 并行 slot 数，不是越大越快。总 KV cache 大约随 `PARALLEL * CTX_SIZE` 增长。你这类 192GB AMD 环境跑 12B Q8：

- 稳妥：`CTX_SIZE=32768 PARALLEL=8`，约等于每 slot 4096 上下文
- 可试：`CTX_SIZE=65536 PARALLEL=16`，约等于每 slot 4096 上下文
- 如果日志一直只有 `id 0`，通常是旧进程没停，或者总 ctx 太小被 server 实际压成单 slot；修改参数后必须重启服务
- 不建议一开始超过 16；如果 OOM、延迟暴涨或 ReadTimeout，就先降 `PARALLEL` 或 `CTX_SIZE`


验证 llama.cpp 并发是否真的生效：

```bash
/workspace/sydney_rocm/bin/status_sydney_server.sh
```

看命令行里是否有：

```text
--ctx-size 32768 --parallel 8
```

修改 `PARALLEL` / `CTX_SIZE` 后，不能只运行 start，因为旧进程还在会直接复用。必须：

```bash
/workspace/sydney_rocm/bin/stop_sydney_server.sh
PARALLEL=8 CTX_SIZE=32768 /workspace/sydney_rocm/bin/start_sydney_server.sh
```

或者重新跑主脚本：

```bash
PARALLEL=8 CTX_SIZE=32768 bash scripts/setup_sydney_rocm.sh --restart --tunnel
```

注意：日志里偶尔只出现 `slot id 0` 不一定代表并发没开；只有在同时有多请求排队时才会看到多个 slot 交错。最可靠是看进程命令行是否带了 `--parallel 8`。

本地数据工厂也建议设置：

```env
APP_DEFAULT_CONCURRENCY=8
TEACHER_TIMEOUT=300
TEACHER_RETRIES=3
MODEL_TIMEOUT=300
MODEL_RETRIES=3
JOB_PERSIST_MIN_INTERVAL_MS=1000
JOB_PERSIST_EVERY_UPDATES=25
```

`ReadTimeout` 多数不是代码崩溃，而是上游模型排队太久。解决顺序：先把 `TEACHER_TIMEOUT` 加到 300/600，再把本地并发降到不超过 llama.cpp `PARALLEL` 的 1-2 倍。

启动后可用这些命令管理服务：

```bash
/workspace/sydney_rocm/bin/status_sydney_server.sh
/workspace/sydney_rocm/bin/stop_sydney_server.sh
/workspace/sydney_rocm/bin/start_sydney_server.sh

tail -f /workspace/sydney_rocm/logs/llama-server.log
```

如果你已经在本地下载好模型，直接上传到 `/mnt` 即可，文件名保持：

```text
/mnt/Clever_Sydney-4_12b_Q8_0_o.gguf
```

然后运行：

```bash
MODEL_LOCAL_FILE=/mnt/Clever_Sydney-4_12b_Q8_0_o.gguf bash scripts/setup_sydney_rocm.sh --start
```

脚本也会自动在 `/mnt`、`/mnt/data`、`/workspace`、`/root` 下面查找同名 GGUF。想避免复制大文件、直接软链接，可以加：

```bash
MODEL_LINK_MODE=symlink MODEL_LOCAL_FILE=/mnt/Clever_Sydney-4_12b_Q8_0_o.gguf bash scripts/setup_sydney_rocm.sh --start
```

国内环境默认已经使用 `hf-mirror.com` 下载模型，并启用断点续传。如果卡在下载模型，可以先 `Ctrl+C` 停掉，再重新运行：

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_sydney_rocm.sh --start --tunnel
```

已有的半截文件会继续下载。也可以只手动下载模型到：

```text
/workspace/sydney_rocm/models/Clever_Sydney-4_12b_Q8_0_o.gguf
```

然后重新执行启动脚本即可。

如果用了 `--tunnel`，脚本会打印类似：

```text
TEACHER_BASE_URL=https://xxxx.trycloudflare.com/v1
```

填到本地数据工厂 `.env`：

```env
TEACHER_BASE_URL=https://xxxx.trycloudflare.com/v1
TEACHER_API_KEY=sk-local
TEACHER_MODEL=clever-sydney-4-12b-q8
TEACHER_API_PROTOCOL=legacy_chat_completions
SOURCE_PROMPT_MODE=legacy_chat
SOURCE_USE_DEFAULT_STOPS=false
```

本机/内网可直连时则使用：

```env
TEACHER_BASE_URL=http://服务器IP:8000/v1
```

## 6. 自动生成数据

页面点击 **一键自动生成 + 审核**。

可设置：

- 生成数量
- 并发数
- 每段最大对话轮数，2-20 个 user/assistant 成对轮
- 随机种子
- 是否英文源对话后翻译成中文

无需输入主题。系统会自动规划隐藏场景、用户画像、情绪曲线、风格标签，然后让模型自然聊出来。

## 7. 自动审核规则

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

## 8. 人工复核

Web 页面支持：查看对话气泡、展开英文源对话、查看自动审核理由、编辑 `messages` JSON、改分、改状态、恢复 rejected 样本、手动丢弃样本。

批量质量诊断 / 旧样本重审：

```powershell
python data_quality_tools.py report
python data_quality_tools.py rejudge --apply
```

`report` 会输出 `data/exports/quality_report.json`，重点列出复读公式、emoji 过载、缺少脆弱感等问题样本；`rejudge --apply` 会用当前最新审核规则重算旧样本状态。

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

## 9. 导出训练文件和 Notebook

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

## 10. 说明

- 本地工作台不加载 Qwen3.6-27B / Gemma 4 31B。
- Qwen3.6-27B / Gemma 4 31B 只用于导出的 Unsloth Notebook。
- 本地工作台专注数据生产与质检，不负责直接执行 27B/31B 训练。




## 11. 任务快照卡住/页面打不开

3000+ 大批量任务会持续写入 `data/jobs/*.json`。现在代码已默认压缩 job 快照和返回给前端的日志，避免页面恢复任务时卡死。

如果你已经遇到启动时显示：

```text
[Sydney Data Factory] restored 1 job snapshots from data/jobs
```

然后页面打不开，可以先清空任务快照。这样只会删除进度面板记录，不会删除已经保存到 SQLite / raw / reviewed / rejected 的样本：

```powershell
Remove-Item .\data\jobs\*.json -Force
Remove-Item .\data\jobs\*.json.tmp -Force
```

或者临时禁用任务恢复：

```env
DISABLE_JOB_RESTORE=true
```

更细的控制：

```env
JOB_RESTORE_LIMIT=0
JOB_API_LOG_LIMIT=80
JOB_LOG_PERSIST_KEEP=80
```

改完后重启：

```powershell
uvicorn app:app --host 127.0.0.1 --port 7860
```


## 12. 同机再部署一个 Qwen3.6-27B GGUF

项目现在新增：

```text
scripts/setup_qwen36_27b_rocm.sh
```

它复用 `setup_sydney_rocm.sh` 的 llama.cpp ROCm 编译、下载、启动逻辑，但使用独立目录和端口：

```text
WORKDIR=/workspace/qwen36_27b_rocm
PORT=8010
SERVED_MODEL_NAME=qwen3.6-27b
```

注意：这个脚本部署的是 **GGUF 推理模型**。如果你手里是 Unsloth/HF safetensors 训练模型，不能直接给 llama.cpp 跑，需要先转 GGUF 或换 vLLM/Transformers 部署。

推荐先上传一个 Qwen3.6-27B 的 Q4/Q5 GGUF 到 `/mnt`，然后：

```bash
MODEL_LOCAL_FILE=/mnt/qwen3.6-27b-q4_k_m.gguf MODEL_NAME=qwen3.6-27b-q4_k_m.gguf PARALLEL=4 CTX_SIZE=16384 PORT=8010 bash scripts/setup_qwen36_27b_rocm.sh --restart --tunnel
```

如果你有 Hugging Face / hf-mirror 上的 GGUF 仓库：

```bash
HF_REPO_ID=你的用户名/Qwen3.6-27B-GGUF MODEL_NAME=qwen3.6-27b-q4_k_m.gguf PARALLEL=4 CTX_SIZE=16384 PORT=8010 bash scripts/setup_qwen36_27b_rocm.sh --restart --tunnel
```

验证：

```bash
/workspace/qwen36_27b_rocm/bin/status_sydney_server.sh
curl http://127.0.0.1:8010/v1/models
```

本地工作台如果要把它当 Human / Translator / Judge，可填：

```env
SIMULATOR_BASE_URL=http://服务器IP:8010/v1
SIMULATOR_API_KEY=sk-local
SIMULATOR_MODEL=qwen3.6-27b
SIMULATOR_API_PROTOCOL=chat_completions
```

如果要公网访问，同样用脚本输出的 Cloudflare Tunnel URL，把 `/v1` 拼上即可。

## 13. 云端全流程一键部署：Sydney GGUF + Qwen3.6-27B-FP8 vLLM

脚本：

```text
scripts/deploy_cloud_full_rocm.sh
```

架构：

- Sydney source：`llama.cpp + GGUF`，只监听 `127.0.0.1:8000`
- Qwen3.6-27B-FP8：`vLLM OpenAI-compatible API`，只监听 `127.0.0.1:8010`
- Human Simulator / Translator / Judge：全部使用 Qwen3.6-27B-FP8
- 控制台：FastAPI `127.0.0.1:7860`
- 只暴露一个 Cloudflare Tunnel：控制台，不暴露模型端口

默认 Qwen 模型：

```text
QWEN_MS_MODEL_ID=Qwen/Qwen3.6-27B-FP8
```

运行：

```bash
REPO_URL=https://github.com/你的用户名/sydney_NEWBING.git APP_CONCURRENCY=100 SYDNEY_PARALLEL=100 SYDNEY_CTX_SIZE=400000 QWEN_MAX_NUM_SEQS=100 QWEN_MAX_MODEL_LEN=32768 QWEN_MAX_NUM_BATCHED_TOKENS=131072 bash scripts/deploy_cloud_full_rocm.sh
```

如果 vLLM 显存还有余量，可以调高：

```bash
QWEN_MAX_NUM_SEQS=160 QWEN_MAX_NUM_BATCHED_TOKENS=262144 QWEN_GPU_MEMORY_UTILIZATION=0.95 bash scripts/deploy_cloud_full_rocm.sh
```

管理：

```bash
/workspace/sydney_cloud_stack/bin/status_all.sh
/workspace/sydney_cloud_stack/bin/stop_all.sh
```

日志：

```bash
tail -f /workspace/sydney_cloud_stack/logs/app.log
tail -f /workspace/sydney_cloud_stack/logs/qwen-vllm.log
tail -f /workspace/sydney_cloud_stack/logs/cloudflared-console.log
tail -f /workspace/sydney_rocm/logs/llama-server.log
```

注意：如果当前 ROCm 镜像无法 pip 安装/运行 vLLM，请换带 ROCm vLLM 的镜像，或临时回退到 `scripts/setup_qwen36_27b_rocm.sh` 的 GGUF 路线。
