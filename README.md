# Multimodal Raw-RAG AML

面向 Agent Memory Leaderboard Add/Search 契约的多模态 Raw-RAG 适配。
本仓库派生自 [Litmeb/RAW-RAG](https://github.com/Litmeb/RAW-RAG)，上游基线提交为
`95639191e9547219b426710e50fe1a36235e26fc`，保留原项目 LICENSE。
原方法与论文作者归上游项目所有；本仓库并非原作者官方发布的比赛服务。
原作者、论文及逐项方法改动见 [来源披露](methods/EverCore/evaluation/docs/METHOD_DISCLOSURE_ZH.md)。

## 说明

提交比赛所用方法中的gpt-4o-mini来自中转站https://yinli.one

## 方法

- 原始历史切片和图片描述索引 → 向量召回 → 一次候选重排（保留全部候选排序）→ 展开完整原始消息 → 按消息 ID 去重 → 取前 15 条来源消息，返回原始文字与图片；Search 不生成最终答案。
- Add 保留传入的原文与 caption；每张历史图片额外通过一次 `gpt-4o-mini` 调用生成最多 4 条互补细节描述。原始相邻文本及各条生成描述独立可检索，并绑定对应图片和原始消息；不把生成描述改写进原始证据。
- 含图 Search 先描述查询图片，再用原始问题和图片描述独立向量召回，以 RRF 合并去重，之后只重排一次；当前不使用直接图片向量检索。
- 默认模型：`gpt-4o-mini`、`text-embedding-v4`、`rerank-v3.5`。
- 窗口 256 tokens，重叠 32 tokens；每路召回最多 128 个候选，融合后最多重排 128 个候选，展开去重后再取最多 min(top_k, 15) 条消息；重复或超媒体预算时继续向后补足。
- 比赛接口按 session 切片、按 user_id 隔离，支持同用户跨会话检索。
- 新增 SQLite 持久化、幂等写入、访问认证及媒体校验。
- 当前代码为 `v1.3`，不沿用历史版本成绩；实际部署版本以 `/health` 为准。图片多描述及噪声控制见 [v1.3 说明](methods/EverCore/evaluation/docs/MULTI_VIEW_IMAGE_RETRIEVAL_ZH.md)。双路查询基础见 [v1.2 说明](methods/EverCore/evaluation/docs/PROVIDED_CAPTION_DUAL_QUERY_ZH.md)。ATM 来源兼容范围及消息去重规则见 [核查说明](methods/EverCore/evaluation/docs/ATM_AND_MESSAGE_DEDUP_ZH.md)。


## 部署

需要 Linux、Python 3.10+。默认通过外部 API 调用模型，无需在接口服务器加载大模型。

```bash
cd methods/EverCore
python3 -m venv .venv
.venv/bin/pip install -r requirements-aml.txt
cp .env.aml.example .env.aml
chmod 600 .env.aml
```

在 `.env.aml` 填写模型地址、模型 Key，以及至少 32 字符的随机 `AML_API_TOKEN`。
`AML_API_TOKEN` 是本服务的访问密码。
如重排需要代理，可设置 `RERANK_PROXY_URL`；请确保网络出口和模型额度持续可用。

v1.3 增加独立图片描述索引，必须使用全新 `--data-dir`，不能直接复用
v1.2 或更早版本的数据库或已有 Full 运行状态。服务会拒绝旧配置的数据目录，不自动删除数据。

```bash
.venv/bin/python -m evaluation.src.aml serve \
  --env-file .env.aml --data-dir /absolute/path/to/private-memory \
  --host 127.0.0.1 --port 8080
```

常驻服务与公网 IP 入口模板见 [deployment](methods/EverCore/deployment/)。
公网使用 HTTPS，配置有效证书、自动续期及必要安全组规则。

## 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 无需认证的健康检查 |
| POST | `/v1/memories/add` | 写入并持久化记忆 |
| POST | `/v1/memories/search` | 返回相关证据 |

认证支持 `Authorization: Token <AML_API_TOKEN>`、Bearer 或 `X-Api-Key`。

**Add 请求：**

```json
{
  "request_id": "write-1",
  "user_id": "user-1",
  "session_id": "session-1",
  "messages": [{"role": "user", "content": "I bought a red card in Paris."}]
}
```

写入完成且立即可检索后返回 HTTP 200，响应为
`{"success": true, "request_id": "write-1", "user_id": "user-1", "session_id": "session-1"}`。
同一用户的相同 request_id 与载荷幂等；同一请求号但载荷不同则返回冲突。

**Search 请求：**

```json
{"user_id": "user-1", "query": "Where did I buy the card?", "top_k": 100}
```

响应为 `{"data": [{"id": "stable-id", "content": "source evidence", "score": 0.9}]}`。
按相关性排序，不超过 `min(top_k, 15)` 条；无结果为 `{"data": []}`。
多模态 content/query 使用有序 `text` / `image_url` 数组，图片必须为内联 Base64 Data URI。
完整格式见 [接口说明](methods/EverCore/evaluation/docs/AML_CONTRACT_ZH.md)
和 [比赛官方契约](https://agentmemoryleaderboard.ai/api-guide)。

## 限制与数据管理

- 图片支持 JPEG、PNG、WebP；单图解码后不超过 10 MiB。
- 单次 Add 和单次 Search 响应的媒体总量各不超过 30 MiB；请求体最大 128 MiB。
- Add/Search 共享实际处理并发 2；超出名额的已认证请求先排队，取得名额后才读取请求体，不因入口繁忙立即返回 429。每个数据目录仅允许一个服务进程。
- 平台建议 Add/Search 并发各设为 1；无法调整平台并发时，服务端仍最多同时处理 2 个请求。排队会增加响应时间，仍受平台及网关的 30 分钟单请求超时限制，不保证任意规模并发均可完成。
- 此入站排队修复不改变 v1.3 的检索、模型、索引身份或返回数量，已有同配置数据目录可继续使用；Full 故障修复应保留变更记录，并使用平台允许的原任务断点恢复。
- 不记录原始 API 请求正文；按赛事要求安排数据清理。
- 停止服务后，可使用 `python -m evaluation.src.aml purge-user --env-file .env.aml --data-dir <目录> --user-id <用户>` 删除指定用户数据。

## 测试

在 `methods/EverCore` 目录执行，无需真实模型 Key：

```bash
pip install pytest pytest-asyncio
python -m pytest tests/test_aml_api.py tests/test_aml_gate.py tests/test_aml_smoke.py -q
```

官方文档样例自测与已知缺口见 [接入复查记录](methods/EverCore/evaluation/docs/AML_AUDIT_20261026_ZH.md)；自测不等于平台正式 Smoke。
