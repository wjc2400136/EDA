# VLM 定量评估协议

[English](VLM_EVALUATION.md) | [简体中文](VLM_EVALUATION.zh-CN.md)

本实验是基于标签条件的经验性图像识别评估，不是 certified robustness 测试。
调用商业或托管 VLM 可能产生较高费用。统计或检查已有结果不需要再次调用服务商；
协议、脚本、模型信息和有效样本数足以复核聚合过程。

## 1. 固定 100 图像子集

从 NIPS 2017 ImageNet-Compatible Dataset 中按文件名进行字典序排序，取前 100
张图像。`data/vlm_first100_manifest.csv` 固定记录：

- 序号；
- 原始文件名；
- ground-truth label 文本；
- target label 文本。

所有方法、VLM 和评估配置必须使用完全相同的 100 个图像身份及标签。目录格式为：

```text
vlm_eval/images/
|-- clean/001_<filename>.png
|-- l2t/001_<filename>.png
|-- bsr/001_<filename>.png
|-- decowa/001_<filename>.png
|-- ops/001_<filename>.png
|-- sid/001_<filename>.png
`-- eda/001_<filename>.png
```

前缀 `001_` 到 `100_` 与 prompt 编号一致。准备脚本不会重新生成对抗样本，只会
从已有目录复制固定子集：

```bash
python experiments/vlm/prepare_vlm_subset.py \
  --manifest ./data/vlm_first100_manifest.csv \
  --clean_dir ./data/images \
  --l2t_dir /path/to/l2t/images \
  --bsr_dir /path/to/bsr/images \
  --decowa_dir /path/to/decowa/images \
  --ops_dir /path/to/ops/images \
  --sid_dir /path/to/sid/images \
  --eda_dir /path/to/eda/images \
  --output_dir ./vlm_eval/images \
  --prompt_file ./vlm_eval/prompts.txt
```

若任意方法缺少固定 100 张中的一张，脚本会失败并列出缺失路径，不会悄悄跳过。

## 2. 固定 Prompt

每次请求只包含当前图像与固定标签条件 prompt。完整 prompt 由
`run_vlm_batch.py` 中的 `build_prompt_text()` 构造，核心任务是判断图像主要视觉
内容是否属于给定真实标签，并且只允许返回：

```json
{"match": 1, "confidence": 8}
```

其中：

- `match=1`：主要对象或场景与标签匹配；
- `match=0`：不匹配；
- `confidence`：1--10 的整数；
- 不允许额外解释性文本。

解析器只容忍代码围栏等格式噪声，不会改变模型的 `match` 或 `confidence`。

## 3. 指标定义

对图像 `i`、VLM `v` 和第 `t` 个评估配置：

```text
s(i,v,t) = match(i,v,t) * confidence(i,v,t) / 10
```

因此：

- 匹配失败时分数为 0；
- 匹配成功时分数在 0.1--1.0；
- 每个 trial 先在有效图像上求平均；
- 再对五个 trial 的均值计算 mean 和 sample standard deviation；
- 跨模型 macro average 应给每个 VLM 相同权重，而不是按响应数量加权。

建议同时报告：

1. 每个模型 clean score；
2. 每种攻击 score；
3. clean-to-adversarial absolute reduction；
4. relative reduction；
5. 有效图像数量；
6. 五次配置的样本标准差；
7. 12 个模型的 macro average。

## 4. API 密钥安全

禁止将真实密钥写入 Python、Markdown、提交到 Git 的 shell 脚本或结果 JSON。
通过环境变量提供：

```bash
export OPENAI_API_KEY='REDACTED'
# 或
export DASHSCOPE_API_KEY='REDACTED'
```

OpenAI-compatible API 示例：

```bash
python experiments/vlm/run_vlm_batch.py \
  --base-url https://provider.example.com/v1 \
  --model PROVIDER_MODEL_ID \
  --image-dir ./vlm_eval/images \
  --prompt-file ./vlm_eval/prompts.txt \
  --results-dir ./outputs/vlm/PROVIDER_MODEL_ID \
  --workers 1
```

应记录 provider、完整 model ID、access route、access date、是否支持 temperature
和 top-p，以及 API 还是隔离的交互任务。

## 5. 无状态请求与五次配置

对于 API 模型，每次调用只携带当前图像和固定 prompt，不保留历史消息、不提供纠错
反馈，也不把前一次输出传给下一次请求。

现有五次评估使用：

```text
temperature = 0.65, 0.80, 0.95, 1.10, 1.25
top-p = 0.9（接口支持时）
```

因此这五次结果是**五个 temperature 配置下的评估**，不是“相同 temperature 下
五个随机种子”的严格重复。准确的描述为：

```text
mean and sample standard deviation across five temperature-varying evaluations
```

或：

```text
variation across five evaluation configurations
```

不应声称该标准差只反映模型采样随机性。

对于不开放采样参数的隔离任务，unsupported parameters 使用 provider defaults。
汇总和解释结果时必须透明披露这一限制。

## 6. 原始结果文件

严格记录格式为：

```text
outputs/vlm/MODEL_ID/
`-- prompt_001/
    `-- eda/
        |-- trial_01.json
        |-- trial_02.json
        |-- trial_03.json
        |-- trial_04.json
        `-- trial_05.json
```

每个 JSON 保存：prompt 编号、文件名、方法、trial、model ID、base URL、标签、
temperature、top-p、完整 prompt 和解析后的响应。共享或汇总结果前检查 JSON 中没有
API key、用户主目录或其他个人信息。

## 7. Provider 拒绝和缺失响应

如果 provider 返回 `data_inspection_failed` 或其他服务端拒绝：

- 不更换图像；
- 不伪造或插补 `match=0`；
- 保留失败记录；
- 明确标注受影响的模型、prompt 编号和错误；
- 每列按成功评估的图像数量计算，并报告有效分母；
- clean 和不同攻击的分母不一致时必须单独说明。

例如 Prompt #093 在某些接口被 provider 拒绝时，该模型列应报告 99 个有效样本，
而不是默认仍为 100。

## 8. 结果汇总

完成所有可用模型后运行：

```bash
python experiments/vlm/summarize_vlm_results.py \
  --model-result MODEL_A=./outputs/vlm/MODEL_A \
  --model-result MODEL_B=./outputs/vlm/MODEL_B \
  --output ./outputs/vlm/vlm_summary.csv
```

每增加一个模型就重复一个 `--model-result NAME=DIR`。输出包含：

- `model`；
- `method`；
- `mean_percent`；
- `sample_std_percent`；
- 五次评估中的最小/最大有效图像数；
- 有效 trial 数；
- `MACRO_AVERAGE` 行。

汇总脚本优先读取 strict per-trial JSON，也兼容旧的每方法 aggregate JSON。

## 9. 结果解释边界

可以支持的结论：

- VLM 实验由定性案例扩展为固定协议下的定量评估；
- EDA 在多数测试模型上产生更大的 label-conditioned recognition degradation；
- 结论在五个 temperature 配置下总体一致；
- 结果同时报告 clean baseline、模型内下降和 macro average。

不应支持的结论：

- 所有 VLM 都普遍脆弱；
- 五次结果是严格相同条件下的独立随机重复；
- 交互界面和 API 的采样控制完全一致；
- 缺失响应可以视为攻击成功；
- 该实验代表 certified 或理论鲁棒性证明。
