# 系统服务与实时精修

system/ 包含本地 Refiner、ASR 后端和 Web 应用。浏览器实际使用时，Qwen 或 Whisper ASR 后端与 Web/Refiner 是独立服务；Web 前端接收音频或转写更新，Refiner 对文本执行保护、精修、校验和失败回退。

## 当前仓库启动方式

在仓库根目录的两个终端分别执行：

~~~bash
bash scripts/start_qwen_asr.sh
bash scripts/start_web.sh
~~~

启动后默认打开 http://127.0.0.1:8081。Qwen 服务默认使用 8766 端口。脚本针对当前工作机设置了 Conda 环境、模型目录和 GPU 编号；更换环境请通过脚本读取的环境变量覆盖，不能假设这些绝对路径适用于其他机器。ASR 端需要 Qwen 模型与 Silero VAD 文件；Web 端需要 Refiner checkpoint 与 SQLite 术语库。

若使用 Whisper，启动 system.whisper_stream_server 代替 Qwen 流式服务，再把 Web 的 ASR URL 指向该服务。各服务参数、VAD、术语库和规则说明见下方原有详细章节。

## 当前实现边界

- 浏览器支持麦克风在线、音频文件整段离线、音频文件分块流式三种处理模式。
- Web 精修入口当前有 off / conservative / tri_state 门控、实体准备与掩码恢复、模型数字规范化、固定数字成语保护、输出校验、重试和回退。
- `system/quantifiers.py` 提供广覆盖量词安全词表；确定性去重保护“数词/限定词 + 量词 AA”（如 `一朵朵`、`一层层`），并以结构模式兜底未收录的量词。
- 文件流式精修采用 latest-only 调度。所有门控模式共用 `fixed_k3_context_v2`：从 ASR/VAD 已完成片段构造标点块，以 K=3 固定分组，每窗增加前后各一块只读原文。流式阶段 `ABC` 等到 `D` 完成才精修，`DEF` 等到 `G` 完成；终态直接处理剩余尾部。模型的用户消息只有当前窗口，上下文放在系统提示中，以兼容现有 Refiner checkpoint。输出和补丁只归属当前窗口；拒绝复制上下文和删除被后窗谓语承接的短主语。缓存按当前分组及前后文复用；任一上下文被 ASR 改写，相关缓存及浏览器显示结果失效。等待不计入门控跳过。tri_state 另外根据 HypothesisTracker 输出 KEEP / DEFER / REFINE。
- 流式阶段不重跑原文未变的拒绝窗口；终态是否重跑只由 `--final-refinement-mode` 决定。仅有数字格式信号的模型调用保留原始窗口末尾标点，这一保护对所有门控模式相同；存在其他语言清理信号时仍可进行受约束的标点修改。
- Qwen 可返回可选 token-logprob 派生分数；未经开发集温度校准前不能当作可靠概率。
- 可选 `--confidence-window-scores` 保留跨 VAD 的源文本分数。Web 的 `--confidence-supplement-mode off|shadow|on` 仅补充送审，不取消原规则精修；缺分 on 保守审核，且不能与学习门控 on 同用。配置及核验边界见 [说明](../CONFIDENCE_SUPPLEMENT_GATE.md)。
- sherpa-onnx + MLX、Qwen 麦克风客户端和浏览器 Web 属于不同运行路径，所需依赖也不同。

当前代码路径和验证边界见 [CURRENT_WORK_AND_NEXT_STEPS.md](../CURRENT_WORK_AND_NEXT_STEPS.md)。

---

## 原有流式系统说明

`system/` contains the streaming implementation used by the AgenticASR desktop App. The packaged Windows/macOS application is distributed through the [VibeXASR product page](https://vibexasr.speech.wiki/); this directory is the reproducible Python implementation.

## Runtime Path

```text
WAV/microphone -> VAD -> online sherpa-onnx ASR -> ChunkManager -> K=3 Refiner
```

The full App path requires a Refiner. The current backend is MLX-LM on macOS. `--identity-refiner` is only an ASR/chunking diagnostic mode.

## Models

Install dependencies:

```bash
python -m pip install -r system/requirements.txt
python -m pip install mlx mlx-lm
```

The browser Qwen backend also needs `sherpa-onnx` in the environment that
starts `qwen_asr_stream_server`; it provides the Silero VAD runtime:

```bash
python -m pip install sherpa-onnx
```

Place an online sherpa-onnx checkpoint under `models/asr/` with `tokens.txt` and either transducer files (`encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`) or a Wenet CTC model (`model*.onnx`).

Download the default Silero VAD:

```bash
bash system/download_vad.sh models
```

Other supported modes are `--vad energy` (no model) and `--vad firered` (install `fireredvad` and provide `--firered-dir`).

Convert a trained Refiner for the current Mac backend:

```bash
python pipeline/scripts/convert_to_mlx.py \
  --input /path/to/huggingface-refiner \
  --output models/refiner-mlx \
  --quantize q4_0
```

Run:

```bash
python -m system.live_asr \
  --wav path/to/example.wav \
  --asr-dir models/asr \
  --refiner models/refiner-mlx
```

## Selectable ASR backends for the browser frontend

The browser frontend is ASR-backend agnostic. It talks to a streaming HTTP
contract (`/stream/start`, `/stream/chunk`, and `/stream/finish`) and applies
the same local Refiner to whatever text the backend returns.

The web UI provides one microphone mode and two local-file modes. Online mode
uses the microphone and refines changed ASR hypotheses synchronously as speech
arrives. Offline mode selects a local audio file, transcribes the entire file,
then runs one final Refiner pass. Streaming mode also selects a local audio
file, sends it in PCM chunks, and refines only the latest pending hypothesis in
a background task so Refiner latency does not block transcription updates.

Qwen3-ASR is the low-latency online backend:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-asr \
  python -m system.qwen_asr_stream_server \
  --model /path/to/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 \
  --confidence-logprobs 5 \
  --vad silero \
  --vad-model models/silero_vad.onnx \
  --vad-min-silence 0.7 \
  --vad-min-speech 0.25 \
  --vad-preroll 0.35 \
  --silence-rms-threshold 0.002 \
  --port 8766
```

`--confidence-logprobs 5` enables an experimental vLLM token-logprob path.
The service then returns `confidence`, `confidence_mean_logprob`, and related
metadata on streaming responses. The score is a geometric mean token
probability and is **not calibrated** until a development-set temperature is
learned. Use `--confidence-temperature T --confidence-calibrated` only after
that calibration. Until then, treat the score as a shadow/experimental signal;
the current conservative gate may consume it, so do not interpret skipped
segments as a production-quality decision.

The default browser path uses a per-session Silero VAD before Qwen decoding.
Non-speech chunks are returned as `silence_skipped=true` and are not sent to
ASR or Refiner. A short pre-roll (`--vad-preroll`) is replayed when speech
starts, and the endpoint chunk is retained so that the first and last syllables
are not dropped. The RMS check remains a cheap fallback when `--vad off` is
selected; set `--silence-rms-threshold 0` to disable that fallback. For a
dependency-free diagnostic, use `--vad energy` instead of `--vad silero`.

The service bounds each Qwen streaming state to protect long recordings from
the upstream implementation's growing full-audio reprocessing cost. It starts
a fresh state at a sentence boundary after 30 seconds and always rotates by 45
seconds, while preserving the cumulative transcript. Use `--segment-seconds`
and `--max-segment-seconds` to tune these limits. Abandoned browser sessions
are cancelled so their queued chunks do not continue occupying the GPU.

Whisper is also supported through a rolling-window backend. Whisper is not an
online transducer, so it re-transcribes the accumulated utterance every few
seconds; this is more compute-heavy but uses the same browser UI:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.whisper_stream_server \
  --model /path/to/whisper-model \
  --device cuda:0 \
  --language Chinese \
  --update-interval-seconds 1.5 \
  --port 8766
```

Start the frontend against either backend:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:0 \
  --asr-url http://127.0.0.1:8766 \
  --language Chinese \
  --port 8081
```

The Whisper backend accepts a local Hugging Face Whisper directory or a model
id such as `openai/whisper-small`; use a local directory for offline runs.

## Protected entity database and session memory

The frontend audits numbers, dates, email addresses, URLs, identifiers, and
verified domain entities alongside each Refiner request. When a verified alias
is explicitly present in an ASR hypothesis, its canonical form is supplied as
a short Refiner glossary. For an entity using `normalize`, that exact alias is
then deterministically normalized in the Refiner output and recorded in the
session log. This does not use fuzzy global replacement: an unrelated or
unmatched database entry cannot change the displayed text.

Create or update the local SQLite term database:

```bash
python -m system.manage_entities --db data/entities.db add AgenticASR \
  --alias "Agentic SR" \
  --type PROJECT \
  --policy normalize \
  --priority 10

python -m system.manage_entities --db data/entities.db add Qwen3-ASR \
  --alias "Qwen ASR" \
  --type MODEL \
  --policy normalize

python -m system.manage_entities --db data/entities.db list
```

`preserve` keeps the exact matched surface. `normalize` restores the verified
canonical spelling when an explicit alias is matched. Fuzzy matching has four
server modes: `off`, `shadow`, `hint`, and `auto`. The default is `shadow`,
which records candidates without changing output. Matching config v2 checks
independent Chinese token boundaries and Latin word boundaries. `auto` applies
only harmless width/spacing/case variants directly. Phonetic substitutions
remain unmasked candidates: the Web Refiner receives observed/canonical pairs,
and a substitution requires a unique, bounded JSON patch with the exact pair.
Short Chinese homophones and boundary-conflicting spans stay raw. Verified
aliases still normalize deterministically. Failed candidate edits fall back
to the trusted-alias baseline, and their rejected normalizations are not logged
as applied. `PERSON`/`ORG` candidates require verified aliases.

The browser also offers **词库辅助识别**. It defaults off; selecting it sends
up to 16 canonical names from the selected domain to Qwen's streaming
`context`. General-domain entries are not added to other domains' audio prompts.
The backend acknowledges application and retains the context across VAD/state
rotation; unsupported backends continue transcription and show that the prompt
was not applied. This is an optional contextual prompt, not a hard decoding
constraint. Its impact needs paired real-audio testing.

**实体原音复核** (`--enable-entity-audio-repair`) confirms glossary candidates
against captured audio before publishing bounded entity revisions. Detection
and replacement run during streaming. For unfinished ASR segments, candidates
need sufficient stable right context; completed segments do not have that
extra requirement.

At finish, recordings of at most 16 seconds also scan the final transcript
and can run one additional prompted audio pass, with at most eight terms and
a four-second total deadline. This covers names appearing only at the end.
The pass retains the original PCM and only copies confirmed entity changes;
failed, ambiguous, echoed or timed-out reviews keep the original transcript.
Candidates already acoustically reviewed are not retried. Long recordings
retain the existing streaming-only finish policy. The controller options
`short_finish_max_seconds` and `short_finish_wait` configure the limits;
setting either to zero disables the extra pass. The audio/cache window can
reduce the duration limit. Incremental status includes `short_finish_checks`,
`short_finish_changes`, `short_finish_timed_out`, `short_finish_errors`,
`short_finish_wall_seconds` and `short_finish_replay_seconds`.

With entity audio repair enabled, a **direct glossary spelling** route first
normalizes complete Chinese homophones to the user's `normalize` spelling.
It checks tone as well as syllables, so `张炜` can become `张伟` while `背景`
does not become `北京`. Independent lexical and inferred-name boundaries
protect words such as `这张位置图` and longer names such as `张炜明`.
Complete registered longer terms, including `preserve` entries, take
precedence over shorter entries. Multiple competing glossary spellings
abstain. The glossary is never inserted into the boundary tokenizer.

For names of at least three characters, a separate near-sound rule permits
one letter difference in exactly one syllable, with equal character length
and at least one shared character. It does not insert, delete, or reorder
syllables. Other uncertain candidates continue through audio review.
This route publishes `GLOSSARY_SPELLING` changes with `exact_sound` or
`near_sound` provenance; `incremental.glossary_spelling_committed` reports
the currently retained count. It does not require CTC or a new Qwen call.
Trailing names wait for right context or EOS; boundaries are rechecked as
the raw hypothesis changes, so a partial patch can be withdrawn.

Mixed Chinese/Latin glossary entries also support literal spelling
normalization (`mixed_form`): `B一细胞` / `b1细胞` / `Ｂ１细胞` can match
`B1细胞`. Spaces between separately spelled acronym letters are ignored,
so a registered `TCR受体` alias can match `T C R受体`. Only code-local
Chinese integers up to 99, or up to three separately spoken digits, are
normalized; numeric values are not changed. Commas, other punctuation,
ordinary English word spaces, and adjacent alphanumeric characters remain
boundaries. Thus `ILC1` does not match inside `XILC1` or `ILC10`.

Different standard names must be registered as equivalent aliases. For
example, `T细胞受体` can declare `TCR受体`; `嗜中性粒细胞` can declare
`中性粒细胞`. The latter lets the same near-sound rule correct
`中心粒细胞` through the complete shorter surface. This is alias-based
normalization, not permission to insert an arbitrary missing character.
No `LC一`→`ILC1`, `FAST`→`FAS`, or Greek-letter reading is guessed without
an appropriate declared surface/rule. The quality-diagnostic runner loads
aliases through the same `EntityStore` loader as the application.

Short uppercase codes have an opt-in `context_code_indel` route. A matching
`spelling_context` must set `allow_near_spelling=True` and reach its declared
keyword score within 192 characters on either side of the source span.
Only one Latin letter insertion/deletion is allowed; numeric suffixes must
match exactly after code-local normalization. For example, configured ILC
subtypes can complete `LC一` to `ILC1` in lymphocyte context, and configured
FAS can correct `FAST` with independent protein-name context such as APO-1.
The rule rejects lowercase words, competing glossary owners, registered
literal codes and interiors of longer codes/registered phrases. It does not
change `FAST` globally, infer letter substitutions or change subtype numbers.
This permission and its context are per-entry library configuration; existing
libraries do not silently enable it. `set_spelling_context` persists it.

The optional **incremental CTC entity sidecar** runs beside the unchanged
Qwen primary and the audio review. `--entity-ctc-mode=shadow` records acoustic
proposals without applying CTC edits; direct glossary spelling and audio
review remain active. `on` can publish a bounded, source-anchored
entity patch. A missing CTC hit never vetoes the existing audio review. For
short recordings, an audio-review confirmation can replace an overlapping
provisional CTC patch. The session status under
`entity_audio_repair.incremental.ctc` reports submitted audio, acoustic
events, proposals, committed patches, review skips, errors and timeouts.

The sidecar needs a local 16-kHz Chinese streaming Zipformer2 CTC ONNX model
directory containing `model.int8.onnx` and `tokens.txt`. Install the isolated
CPU dependencies in a venv based on `agentic-asr`:

```bash
/home/aim0/data/conda/envs/agentic-asr/bin/python -m venv --system-site-packages .runtime/entity_ctc
.runtime/entity_ctc/bin/python -m pip install -r requirements-entity-ctc.txt
ENTITY_CTC_MODE=shadow \
ENTITY_DB=/absolute/path/to/your/entities.db \
ENTITY_CTC_MODEL_DIR="$PWD/results/entity_backend_comparison_20261015/models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30" \
WEB_PORT=8082 bash scripts/start_web.sh
```

Use `ENTITY_CTC_MODE=on` only for a deliberate CTC-enabled server. Sessions
can request `?entity_ctc=off`, `shadow` or `on` when a model is configured;
CTC requires entity audio repair and is off by default. The model is loaded
once; each session has separate cached acoustic state. The worker processes
new PCM in order on CPU, while the Qwen request continues. Finish drains CTC
once, then applies final evidence before the short-audio review. The selected
`-2` gap and `0.1` margin came from the frozen 171-sample development split;
they are ranking thresholds, not calibrated correctness probabilities.
At most eight CTC sessions are admitted per Web process; additional sessions
keep the original audio-review path. The CTC queue is capped at eight seconds
of unprocessed audio per session;
an overloaded or invalid CTC stream falls back to the existing audio-review
path and reports the reason in the CTC status. The start script also checks
that the configured Python environment can load both CPU dependencies.
The 37 challenge scene libraries used for the reported accuracy comparison
are separate from the Web service's `ENTITY_DB`. Load the entities that belong
to the intended session into that database before expecting the benchmark's
entity coverage; the service never imports benchmark labels or libraries
automatically.
The current implementation imports the validated frontend and spotter from
`experiments/scripts`; package those modules with the Web service when
deploying outside this repository. The comparative results and known-scene
limits are recorded in `results/entity_backend_comparison_20261015/CONCLUSION.md`.

Use `--enable-entity-asr-context` or `ENTITY_ASR_CONTEXT_ENABLED=1` to default
new sessions to enabled; the browser toggle can opt out. The context accepts
only canonical spellings, is bounded to 1024 characters from the Web (2048 at
the ASR API), and does not include observed spelling errors or transcripts.

Enable the database in the browser frontend:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentic-asr \
  python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:0 \
  --asr-url http://127.0.0.1:8766 \
  --entity-db data/entities.db \
  --entity-fuzzy-mode shadow \
  --refinement-gate-mode conservative \
  --output results/web/session.jsonl \
  --port 8081
```

The routing gate is a separate deterministic module and does not change entity
normalization or the post-generation safety validator. Use
`--refinement-gate-mode off` for the always-refine baseline with the same fixed K=3 windows, and use
`--refinement-gate-mode conservative` for the gated experiment. Conservative
mode skips complete high-confidence segments without
cleanup signals. Missing ASR confidence fails open to the Refiner. Each Web
response and JSONL record includes `refiner_executed`,
`refinement_gate_config`, `refinement_gate_decisions`, and
`refinement_gate_skipped_segments`.

The default Web numeric path preserves the prior Refiner validation, safe
numeric-edit salvage, and Python fallback behavior. To try the conservative
zh-itn rule pack before Refiner, set `ZH_ITN_ENABLED=1` when starting
`scripts/start_web.sh`; its native bridge is built with a C++17 compiler if
needed. Set `ZH_ITN_LIBRARY` to use a custom library. Direct startup accepts
`--enable-zh-itn` and `--zh-itn-library`. The response reports
`zh_itn_enabled: true` and `numeric_backend: zh-itn` when enabled, and
`numeric_normalizations` lists its replacements. Some spoken forms remain
unchanged under this conservative rule pack. In zh-itn mode a rejected model result falls back to the zh-itn
baseline. Fixed expressions such as `一五一十` and `三番五次` are protected
from model edits, as are verified terms, acronyms and URLs.

In zh-itn mode, repetition signals entirely inside a converted numeric source
span (for example repeated telephone digits) do not trigger the Refiner.
Repetition outside that span and repeated complete clauses still do. Equal
clock forms such as `9点30分` and `9:30` are aligned and restored to the rule
spelling before validation. Explicit corrections including `哦不` may drop
the superseded clock; unrelated value changes remain rejected. Numeric
integrity failures use the existing safe fallback without a model retry,
unless a protected placeholder was also damaged. Other targeted integrity
retries retain their existing policy.

After sentence windows are joined, a deterministic repetition pass collapses
two or more adjacent identical short utterances (up to eight visible
characters), including exclamation-mark-separated runs such as
`可恶！可恶！`, to one occurrence.
Adjacent repeated pronouns and demonstratives used as stutters, such as
`你你竟结成了元婴` or `我我不知道`, are also collapsed deterministically.
Normal lexical or verb reduplication (`人人`, `天天`, `好好`, `看看`) is preserved.

When the server is started with `--entity-db`, the browser page also exposes a
local **术语库管理** panel. It can search, add, edit, enable/disable, and
permanently delete verified entities and aliases. Changes are stored only in
the configured SQLite file and are loaded by the next transcription session;
they do not interrupt a running session or force a rewrite of displayed text.

Entity types and domains are fixed selectable categories in the panel. The
main page also provides a **术语领域** selector. A new session loads entries in
the chosen domain plus entries in `general`; for example, selecting `医疗`
loads `医疗` and `通用` entities. This selection only changes entity auditing,
the Refiner glossary, and verified-alias normalization; it does not change the
ASR model itself.

Each WebSocket connection also gets a bounded in-memory entity memory.
Database entities are immediately trusted. A newly observed entity is not
allowed to constrain later text unless it has multiple independent
high-confidence observations with distinct stable-segment IDs. Repeated
partial hypotheses do not count as independent evidence, and missing ASR
confidence never promotes an entity automatically. This avoids turning one
ASR or Refiner error into persistent session memory.

## Local Qwen3-ASR + Refiner microphone mode

The original `live_asr.py` requires a sherpa-onnx online ASR model and an MLX
Refiner. For a Linux machine with a local Qwen3-ASR checkpoint and a Hugging
Face Refiner checkpoint, run the two processes below in separate environments.
They communicate only through `127.0.0.1`.

Start the Qwen3-ASR service in an environment with `qwen-asr==0.0.6` and
`transformers==4.57.6`:

```bash
python system/qwen_asr_server.py \
  --model /path/to/Qwen3-ASR-0.6B \
  --device cuda:0
```

In a second terminal, install microphone support in the Refiner environment:

```bash
python -m pip install sounddevice
```

Then start the microphone client in that environment:

```bash
python -m system.live_qwen_refiner \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:1 \
  --entity-db data/entities.db \
  --entity-domain general \
  --entity-fuzzy-mode shadow \
  --output results/live/session.jsonl
```

After reviewing shadow candidates in the JSONL output, use
`--entity-fuzzy-mode auto` for guarded formatting and candidate handling.
Unconfirmed phonetic substitutions are retained as source text. Candidate scores, decisions, reasons, matcher latency, and
applied normalizations are included in each output record.

The client uses local energy VAD and submits one completed utterance after a
short silence. It prints raw and refined text, and optionally records each
utterance to JSONL. Use `--list-devices` to select a microphone.

## Browser microphone mode for VMs

For a PVE VM, microphone access is usually easier through a browser on the
physical computer. This mode uses Qwen's vLLM streaming backend. Stop an
existing `qwen_asr_server.py` process first, then start the streaming service
in the Qwen environment:

```bash
python system/qwen_asr_stream_server.py \
  --model /path/to/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.75
```

In a second terminal, start this Web service in the Refiner environment:

```bash
python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --refiner-device cuda:1 \
  --output results/web/session.jsonl
```

Keep the service bound to `127.0.0.1`. From the physical computer, forward the
VM port over SSH and open the localhost URL in a browser:

```bash
ssh -L 8081:127.0.0.1:8081 user@vm-address
```

Open `http://localhost:8081`. Browsers permit microphone access on localhost;
opening an HTTP URL at the VM's LAN address generally will not. The browser
records WAV audio locally and sends it through the SSH tunnel, so no VM audio
passthrough is required.

## Files

- `live_asr.py`: terminal entry point and streaming loop.
- `backends.py`: audio input, VAD, and sherpa-onnx adapters.
- `chunking.py`: stable bounded chunks from incremental hypotheses.
- `refiner.py`: Refiner protocol, MLX backend, identity diagnostic backend, and sliding-window session.
- `qwen_asr_server.py`: local Qwen3-ASR offline HTTP service for the separate Qwen environment.
- `qwen_asr_stream_server.py`: local Qwen3-ASR vLLM streaming HTTP service.
- `live_qwen_refiner.py`: microphone VAD and Refiner client for the separate AgenticASR environment.
- `web_app.py`: browser API and local Refiner service for VM use.
- `web/index.html`: browser microphone interface.
- `download_vad.sh`: download the Silero VAD model.
- `requirements.txt`: streaming dependencies.
- `__init__.py`: package exports.
