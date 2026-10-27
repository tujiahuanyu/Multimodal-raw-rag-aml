# v1.2 试验：使用传入 caption 与双路查询

本次改变的是独立 AML Add/Search 服务，旧离线全量实验不在原目录中更新；
从线上 v1.1 升级时使用独立代码与数据目录，保留旧版本以便回滚。
这不是图片向量检索，也不改变平台统一 Answer／Eval。实际效果需要独立对照评测。

## Add：不再生成历史 caption

直接将平台传入的所有 text 原样作为检索文本；在对应位置插入历史图片标识，保留原始图片字节。
例如 `text(caption1), image1, text(caption2), image2` 保持该顺序存储、切片和返回，
不会将整条消息的文字交给 VLM 再为每张图片重新生成一遍说明。

官方 ContentPart 没有独立的 caption 字段，text 也可能是普通会话或结构化元信息。
因此不通过猜测将任意文本强行标为某张图片的独占说明，也不重新排列输入。
若来源明确提供逐图 caption，应将每张 caption 放在对应图片旁边；公开 Mem-Gallery
适配器从每张历史图片的 image_caption 读取说明并采用这种顺序。

图片没有 caption 时仍接受、存储并返回，不凭空补描述；该图没有可用于语义索引的独立
说明，可通过同消息的其他文字或命中该消息的来源关系返回。当前无直接图像向量索引。

## Search：两路召回，一次重排

1. 提取原始 query 中的文字，若有 options 则作为原问题的附加检索上下文。
2. 仅为 query 内图片生成 caption；按用户隔离缓存相同查询图的说明，不使用历史文字或答案生成查询 caption。
3. A 路：用原始问题文字独立生成向量，检索含原文／传入 caption 的历史 chunk。
4. B 路：用查询图片 caption 独立生成向量，检索同一个历史索引。多张查询图的不同说明合成一条 B 路查询，不让图片数量变成多路投票优势。
5. 每路最多 128 个 chunk；按 chunk_id 去重，以 `sum(1/(60+rank))` 融合，保留前 128 个。相同融合分数按首次出现顺序排序，A 路先于 B 路。
6. 用“原始问题＋查询图说明”作为相关性判断条件，对融合后的候选只调用一次重排。
7. 按重排顺序展开完整原始消息，按 message_id 去重；跳过媒体预算超限的完整消息后补足，返回最多 min(top_k,15) 条原始证据。

纯文本 query 只有 A 路；纯图片 query 只有 B 路，不为缺失的问题文字生成空向量。
两路查询文字恰好一致时只召回一次。查询图生成说明失败会明确报错，不静默退回纯文本。
查询图片和新生成的说明用于检索，不伪装成历史证据返回。

注意：这两路是“不同文本查询搜索同一历史索引”，不是一个额外的 caption 专用索引，
更不是 CLIP／SigLIP 图片向量召回。

## 状态隔离

服务版本为 raw-rag-aml-v1.2，索引身份 protocol_version=2。
必须选择新的私有 data-dir；v1.1 的目录不能直接重用。不会自动清空或改写旧数据库。
正式 Full 已受理的固定版本不得在运行过程中替换。

## 公开 Mem-Gallery 功能检查

在 methods/EverCore 下使用已有研究环境：

```bash
python -m evaluation.src.aml.memgallery_smoke prepare \
  --data-root /absolute/path/to/Mem-Gallery \
  --scenario AI_Robotics_Automation_Future_Tech --questions 6 \
  --bundle /absolute/path/to/v12-public-smoke/bundle.json

python -m evaluation.src.aml.memgallery_smoke run \
  --env-file .env.aml --data-dir /absolute/path/to/new-v12-memory \
  --bundle /absolute/path/to/v12-public-smoke/bundle.json \
  --output /absolute/path/to/v12-public-smoke/report.json
```

prepare 使用公开数据集的历史 caption，但不把查询图片的现成 caption 填入 query，
查询说明仍由新服务从图片生成。答案及 clue 不进入 API 请求；clue 只在独立 reference
文件中用于检索后检查。研究测试的短报告不等于平台 Smoke 或完整 LLM-Judge 得分。
run 默认仍使用私有配置中的重排服务，执行前确认额度；本地 Qwen 可通过隔离研究包装
替换 Providers.rerank，不修改正式服务的模型配置。

## 2026-10-27 开发机验证

- 在独立源码／数据目录运行完整回归：171 项通过，20.92 秒；使用替身模型，不代表真实答题质量。
- 对公开 Mem-Gallery 的 AI_Robotics_Automation_Future_Tech 场景进行真实接口检查：370 条来源消息、31 张历史图片，26 批 Add，6 次 Search（3 个文字查询、3 个带图查询）。
- Add 的历史图片描述调用为 0；每个文字查询召回一路，每个含图查询召回两路，6 次 Search 各只重排一次；返回的来源消息 ID 均无重复。
- 该检查使用已有 Qwen3-Reranker-4B，在空闲 GPU 1 上运行，不消耗 Cohere 调用额度；向量和查询图片描述使用现有 API。计数为向量请求 34 次、视觉请求 3 次（包括重试／缓存影响）、本地重排 6 次。
- 6 题中有 5 题完整命中标注线索，另 1 题未命中；这不是答案正确率，也不是 LLM-Judge 分数，不能据此声称新方法已提分。
- 没有修改阿里云部署或正在运行的旧 Mem-Gallery 全量实验，也没有发起平台 Smoke／Full。

官方输入格式依据：[赛事说明](https://agentmemoryleaderboard.ai/competition/#tracks)、
[Add/Search 指南](https://agentmemoryleaderboard.ai/api-guide)。
