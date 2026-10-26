# MemGallery 全量评测：本地 Qwen4B 重排

## 使用范围

此流程在有 NVIDIA CUDA GPU 的研究服务器运行。杭州 2 核 4GB、无 GPU 的 ECS
只运行比赛接口，不在该实例加载 Qwen4B。云端默认 Cohere 配置不因本地实验而改变。

模型为官方 [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)，
不是 0.6B 小版本。2026-10-26 在单张 RTX 3090 上实际加载成功，中英文排序检查通过；
128 个合成候选重排耗时 3.62 秒，峰值已分配显存 7,784.7 MiB。
这些是环境自测，不是完整数据集效果或速度保证。

## 检索方法

1. 用视觉模型生成历史图片和查询图片的文字描述，不使用数据集现成说明。
2. 原始会话文字、图片标识和图片描述组成检索历史；按 256 tokens 切片，重叠 32 tokens。
3. 使用 `text-embedding-v4` 生成向量，召回最多 128 个候选。
4. Qwen4B 在本地读取查询与候选文本，以 yes/no 概率进行相关性排序，保留最多 15 条。
5. Reader 使用检索文字和关联的原始图片回答；每题最多 10 张图片，查询图片优先。

因此本实现仍是“视觉描述辅助文本检索”，不是直接图片向量召回，也不是原生视觉重排。
Qwen 配置使用 BF16、SDPA、每批 4 个候选；超出 8,192-token 输入限制时明确报错，
不静默截断，不在失败时偷偷调用 Cohere。

Qwen 重排不消耗 Cohere 调用额度，但图片描述、向量生成及 Reader 仍调用外部 API，
可能产生费用。不要将本配置的结果标为原始 Cohere 配置的成绩。

## 安装与模型检查

在仓库的 `methods/EverCore` 目录，使用已有研究虚拟环境或单独创建环境：

```bash
pip install -r requirements-local-reranker.txt
export QWEN_RERANK_MODEL_PATH=/absolute/path/to/models/Qwen3-Reranker-4B
CUDA_VISIBLE_DEVICES=0 python -m evaluation.src.multimodal.download_reranker \
  --output "$QWEN_RERANK_MODEL_PATH"
```

下载器从官方 ModelScope 模型仓库取得分词器和 safetensors 权重，不下载并执行远程模型代码。
只有实际推理通过中英文排序检查及 128 候选检查，才生成 `load_test_passed.json`。
失败时检查 `download_status.json` 和终端日志，不要手工伪造验证标记。

将相同的 `QWEN_RERANK_MODEL_PATH` 写入私有 `.env.aml`，并填写视觉与向量 API 配置。
本地重排不需要 Cohere Key。不要将真实 `.env.aml` 提交到 GitHub。

## 全量运行

数据单独从 [Ethan-Bei/Mem-Gallery](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery)
获取并遵守其条款；不要把数据、模型权重或私人评测日志提交到代码仓库。

先生成图片描述缓存：

```bash
python -m evaluation.src.multimodal.full_run --stage captions \
  --data-root /absolute/path/to/Mem-Gallery --env-file .env.aml \
  --config evaluation/config/systems/memgallery_local_qwen4b.yaml \
  --output /absolute/path/to/run --caption-workers 2
```

完整数据包括 20 个场景、1,003 张历史图片、487 张查询图片和 1,711 道题。
图片描述缓存逐张保存，可使用相同目录续跑；不要同时启动两个描述任务写同一缓存。

描述完成后进行评测：

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.src.multimodal.full_run --stage evaluate \
  --data-root /absolute/path/to/Mem-Gallery --env-file .env.aml \
  --config evaluation/config/systems/memgallery_local_qwen4b.yaml \
  --output /absolute/path/to/run --wait-for-prerequisites
```

等待模式最多等待 12 小时，要求全部描述缓存和模型检查标记就绪。如果同一输出目录
存在 `captions.pid`、`reranker-download.pid`，还会检查未完成的前置进程是否已停止。
本地 Qwen 配置无需 `--confirm-full-rerank-quota`；该标志仅用于外部重排的额度确认。

## 查看结果与边界

- `captions.json`：图片描述缓存；同目录 manifest 用于核对描述模型配置。
- `evaluation_job_status.json`：等待、运行、完成或失败状态。
- `evaluation/results.jsonl`：逐题结果，运行中也会持续写入，配置一致时可续跑。
- `evaluation/summary.json`：已完成题目的诊断汇总；不足 1,711 题时不是全量结果。
- `evaluation/official_export/`：全量成功结束后的逐场景输出，仍需按数据集正式评分要求评审。

答案和 clue 仅用于检索完成后的诊断统计，不输入召回、重排或 Reader。
EM/F1 和来源轮次召回率不能替代官方 LLM 评审，也不是比赛平台统一 Answer/Eval 的成绩。
离线图片读取上限 20 MiB；比赛 Add/Search 的单图上限为 10 MiB，二者不可混同。
