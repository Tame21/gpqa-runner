# GPQA 便携评测工具

把整个 `gpqa_runner` 文件夹复制到安装了 **ais_bench** 的环境中即可使用。已附带真实 GPQA 数据，不需要重新下载，也不需要修改 `site-packages` 或 ais_bench 安装目录。

本仓库仅包含工具本身，不依赖 benchmark 源码。可以单独复制到其他机器或任意目录，也可以放在现有项目根目录下，与 `benchmark/` 同级。

脚本使用 Python 3.10+ 标准库；实际推理、评分由当前 Python 环境中的 ais_bench 完成。需要能够访问已启动的 vLLM / OpenAI 兼容聊天服务。

## 快速使用

克隆仓库，并激活安装了 ais_bench 的 Python 环境后运行：

```bash
git clone https://github.com/Tame21/gpqa-runner.git gpqa_runner
cd gpqa_runner
python run_gpqa.py --host-ip 127.0.0.1 --host-port 8989 --model step37
```

如果已直接复制工具文件夹，则进入该目录运行即可：

```bash
cd gpqa_runner
python run_gpqa.py --host-ip 127.0.0.1 --host-port 8989 --model step37
```

默认运行 **GPQA Diamond、0-shot CoT chat、精度评测（推理 + 评分）**。默认参数与仓库中的 `vllm_api_general_chat.py` 一致：端口 `8080`、并发 `1`、`max_out_len=512`、温度 `0.01`、非流式、重试 `2` 次；不传 `--model` 时由 ais_bench 查询服务模型名。

对于需要长推理的模型，建议按模型需要显式调大输出长度，例如：

```bash
python run_gpqa.py --host-port 8989 --model step37 --batch-size 8 --max-out-len 32768
```

先用 5 题检查服务和配置：

```bash
python run_gpqa.py --host-port 8989 --model step37 --num-prompts 5 --debug
```

也可以从任意目录调用脚本的完整路径，不必先 `cd` 到工具目录。脚本通过 `Path(__file__).resolve().parent` 自动读取自身所在目录，并在启动时打印 `Tool directory`。数据、配置、提示词、输出以及工具参数中的相对文件路径都以此目录为基准，不需要填写固定盘符或安装路径。Linux、Windows 使用同一份脚本；实际评测的平台支持取决于已安装的 ais_bench。

## 精度评测后的 acc_len 统计

默认在 GPQA 精度评测（`--mode all`）成功完成后，自动请求同一模型服务的 `/metrics`，打印投机解码接受长度，并将结果保存到工具内的 `outputs/acc_len/acc_len_<时间戳>.json`。例如服务配置了 3 个 speculative tokens：

```bash
python run_gpqa.py --host-port 8989 --model step37 --num-spec 3
```

`--num-spec` 应与服务端配置一致；省略时根据指标的 `position` 标签自动推断。计算方法与提供的统计脚本一致：

```text
acceptance_per_pos[i] = num_accepted_tokens_per_pos[i] / num_drafts
acc_len = 1 + sum(acceptance_per_pos)
```

例如 `num_drafts=100`，三个位置接受的 token 数为 `[80, 40, 20]`，则接受率为 `[0.8, 0.4, 0.2]`，`acc_len=2.4`。此处的 `acc_len` 是原脚本中的 `acceptance_len`，与 GPQA 答题准确率、单条回答的输出长度不同。

统计口径沿用原脚本：**服务启动以来的累计计数**，聚合 `/metrics` 返回的全部相关序列，不是本次 GPQA 的前后差值；历史请求、其他客户端请求及 warmup 都可能包含在内。多个子集一起运行时，在全部结束后采集一次。

默认指标地址随 `--host-ip`、`--host-port` 或 `--url` 变化，代理 URL 的路径前缀会保留。如果指标使用独立端口或路径，可覆盖完整指标地址：

```bash
python run_gpqa.py --host-port 8989 --model step37 --num-spec 3 --metrics-url http://127.0.0.1:9000/metrics --metrics-timeout 10
```

无需安装 `requests` 或 `prometheus_client`，统计使用 Python 标准库实现。读取 `_total` counter 样本时会排除 `_created` 时间戳。无 draft、指标缺失或网络失败时明确提示原因，报告中 `acc_len` 为 `null`，不伪造统计值，也不改变已完成的 GPQA 评测退出状态。`--mode infer` 成功后也会统计；仅评分、可视化、性能模式、dry run、搜索配置或评测失败时不采集。

可通过 `--no-acc-len` 关闭，也可在 `settings.json` 中长期配置：

```json
"acc_len": {
  "enabled": true,
  "num_speculative_tokens": 3,
  "metrics_url": "",
  "timeout": 10
}
```

已有服务也可单独运行统计脚本，两个位置参数沿用原脚本的端口和 NUM_SPEC：

```bash
python acc_len.py 8989 3 --host-ip 127.0.0.1
```

## 修改参数，无须手写 Python 配置

常用参数可直接通过命令行指定：

| 参数 | 对应配置 / 用途 |
| --- | --- |
| `--model` | 服务对外暴露的模型名 |
| `--host-ip` / `--host-port` | 服务 IP、端口；别名 `--host` / `--port` |
| `--url` | 完整服务地址，支持代理前缀；优先于 IP 和端口 |
| `--api-key` | API 密钥；也支持环境变量 `AIS_BENCH_API_KEY` |
| `--abbr` | 结果文件中的模型简称 |
| `--path` | 本地 tokenizer 路径；相对路径以工具目录为基准 |
| `--batch-size` / `--request-rate` | 并发数 / 请求速率 |
| `--max-out-len` / `--retry` | 最大生成 token 数 / 重试次数 |
| `--stream` / `--no-stream` | 开启 / 关闭流式 |
| `--use-timestamp` / `--no-use-timestamp` | 开启 / 关闭时间戳流量模式 |
| `--trust-remote-code` / `--no-trust-remote-code` | tokenizer 加载选项 |
| `--temperature` / `--ignore-eos` / `--no-ignore-eos` | 生成参数 |
| `--generation-kwargs` | 合并额外生成参数，支持 JSON 或 `@文件路径` |
| `--set KEY=VALUE` | 添加 / 覆盖模型配置字段，可重复、可用点号设置嵌套字段 |
| `--subset diamond main extended` | 指定一个或多个子集；`--subset all` 运行三个子集 |
| `--prompt cot` / `--prompt str` | 官方 CoT chat / 普通字符串提示词 |
| `--num-prompts N` | 每个子集只运行前 N 题 |
| `--mode` | `all`、`infer`、`eval`、`viz`、`perf`、`perf_viz` |
| `--dump-eval-details` | 保存逐题评估详情 |
| `--reuse [时间戳]` | 复用最近一次 / 指定时间戳的结果 |
| `--generate-only` / `--dry-run` | 只校验数据、生成配置并打印命令；不访问推理服务 |
| `--acc-len` / `--no-acc-len` | 开启 / 关闭运行后的接受长度统计，默认开启 |
| `--num-spec` / `--num-speculative-tokens` | 服务端 speculative token 数，默认从指标位置推断 |
| `--metrics-url` / `--metrics-timeout` | 指标接口地址 / 请求超时秒数（默认 10） |

例如，服务 URL 可以直接包含 `/v1` 或 `/v1/chat/completions`，工具会规范化地址，避免 ais_bench 再次追加时造成重复路径：

```bash
python run_gpqa.py --url http://127.0.0.1:8989/v1 --model step37 --stream
python run_gpqa.py --model step37 --set generation_kwargs.top_p=0.95 --set generation_kwargs.top_k=20 --set generation_kwargs.seed=42
python run_gpqa.py --model step37 --set generation_kwargs.chat_template_kwargs.enable_thinking=false
```

`--set` 的值优先解析为 JSON（数字、布尔值、数组、对象或 `null`），否则作为普通字符串。它也可添加当前 ais_bench 版本支持的其他模型参数，例如 `--set enable_ssl=true`；参数是否被模型支持由 ais_bench 判断。

为避免终端对 JSON 引号的不同处理，可以在工具目录创建 `generation.json`：

```json
{
  "top_p": 0.95,
  "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

```bash
python run_gpqa.py --generation-kwargs @generation.json
```

长期使用时直接修改本目录的 `settings.json`，然后只需 `python run_gpqa.py`。也可以创建仅含需要覆盖字段的 JSON 文件，通过 `--settings my_server.json` 使用：

```json
{
  "model": {"host_ip": "127.0.0.1", "host_port": 8989, "model": "step37", "batch_size": 8},
  "dataset": {"subsets": ["diamond"]}
}
```

优先级：`settings.json` → `--settings` 指定文件 → 命令行参数 → `--set`。`AIS_BENCH_API_KEY` 覆盖 JSON 中的密钥，`--api-key` 和 `--set api_key=...` 可再覆盖它；`--temperature` / `--ignore-eos` 覆盖 `--generation-kwargs` 中的同名项。命令行覆盖只影响本次生成，不回写 `settings.json`。

`--settings`、`--generation-kwargs @JSON文件` 和 `--path` 中的相对路径统一按工具自身目录解析，也可传绝对路径。例如把 `server.json`、`generation.json` 放进工具目录后，在任何终端目录传入 `--settings server.json --generation-kwargs @generation.json` 都会读取同一套文件。要随工具搬迁的 tokenizer 等文件也应放在工具内并使用相对路径；显式填写的外部绝对路径保持原意。提示词在本目录 `prompts/cot.txt` 和 `prompts/str.txt` 中，可直接修改；保留 `{question}`、`{A}`、`{B}`、`{C}`、`{D}` 占位符及各自的答案格式。

## 配置、数据与结果的位置

```text
gpqa_runner/
├── run_gpqa.py
├── acc_len.py                       # 自动 / 独立统计投机解码接受长度
├── settings.json
├── prompts/
│   ├── cot.txt
│   └── str.txt
├── data/
│   ├── manifest.json                 # 数据来源、条数和 SHA-256
│   └── gpqa/
│       ├── gpqa_diamond.csv           # 198 题
│       ├── gpqa_main.csv              # 448 题
│       ├── gpqa_extended.csv          # 546 题
│       ├── gpqa_experts.csv           # 原包附带的专家元数据，不作为评测子集
│       └── license.txt
├── configs/
│   ├── models/vllm_api/vllm_api_general_chat.py
│   ├── datasets/gpqa/gpqa_local.py     # 运行脚本时生成
│   └── gpqa_benchmark.py              # 本次评测的完整配置，运行脚本时生成
├── outputs/                          # ais_bench 的日志、预测和评分结果
│   └── acc_len/                      # 带时间戳的接受长度报告
└── tests/
```

每次执行脚本都会根据当前参数重新生成以上三个 Python 配置文件，包含对应的 `vllm_api_general_chat.py`。生成的配置使用 ais_bench 支持的完整类型名称字符串。脚本通过当前解释器执行 `python -m ais_bench.benchmark.cli.main <完整配置路径>`，直接传入完整配置，避免与安装目录中的同名配置混淆。

数据路径在每次启动时解析为当前工具目录的绝对路径，供 ais_bench 读取；生成文件中出现的绝对路径是本次自动计算的结果，不是脚本写死的路径。整个目录移动或改名后，照常执行 `run_gpqa.py` 就会按新位置重新生成配置，不需要手工修改旧配置或设置路径环境变量。生成文件属于可覆盖产物，持久调整请修改 `settings.json`、提示词或脚本。不要同时从同一个工具目录启动多次评测；并行评测可以复制多份工具目录。

脚本不会下载数据。数据缺失、CSV 格式错误或参数错误时会明确报错。三个子集存在重叠，`--subset all` 会分别评测，不应把它们当作互不重复的数据合并统计。API 密钥会进入本地生成配置及 ais_bench 的配置快照，分享文件夹前请移除密钥和运行产物。

可将其他 ais_bench 选项放在 `--` 后透传：

```bash
python run_gpqa.py --host-port 8989 -- --num-warmups 0
python run_gpqa.py --mode eval --reuse
```

`eval` / `viz` 需要已有运行结果，复用时保持模型与子集等配置一致。`perf` 模式的 tokenizer 等要求遵循已安装的 ais_bench 版本。只验证配置可以用 `--generate-only`；让 ais_bench 自己执行 dry run 可以用 `python run_gpqa.py -- --dry-run`。

## 数据来源与验证

数据按 benchmark 原有 GPQA 文档指定的 [OpenCompass 数据包](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip) 原样内置，来源项目为 [idavidrein/gpqa](https://github.com/idavidrein/gpqa)，作者 Irving David Rein，遵循 **CC BY 4.0**，原始许可见 `data/gpqa/license.txt`。`manifest.json` 记录原压缩包及各文件校验值。

工具的离线测试不需要安装 ais_bench，也不调用模型服务：

```bash
python -m unittest discover -s tests -v
```

测试覆盖数据完整性、配置覆盖、参数校验、目录迁移、子进程调用，以及 acc_len 计算、指标解析和运行后采集流程。实际模型精度以连接真实服务后的 ais_bench 输出为准。
