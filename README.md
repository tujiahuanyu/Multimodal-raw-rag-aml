# Multimodal Raw-RAG AML

面向 Agent Memory Leaderboard Add/Search 契约的多模态 Raw-RAG 适配。
本仓库派生自 [Litmeb/RAW-RAG](https://github.com/Litmeb/RAW-RAG)，上游基线提交为
`95639191e9547219b426710e50fe1a36235e26fc`，保留原项目 LICENSE。
原方法与论文作者归上游项目所有；本仓库并非原作者官方发布的比赛服务。

## 方法与边界

- 图片经视觉模型生成描述，描述与原始会话文字共同用于检索。
- 文本向量召回后重排，返回来源文字和原始图片；当前没有直接图片向量检索分支。
- 默认学术配置使用 `gpt-4o-mini`、`text-embedding-v4`、`rerank-v3.5`。
- 256-token 窗口、32-token 重叠、最多召回 128 个候选、重排返回最多 15 条。
- AML 服务只完成 Add/Search，不在 Search 中生成最终答案。
- MemGallery 离线研究流程另带 Reader 及诊断 EM/F1，不能混同为赛事统一 Answer/Eval。

## 新增实现

1. MemGallery 原始历史、查询图片和参考标注分离加载；来源图片保持可追溯。
2. 视觉描述检索及多模态证据返回。
3. SQLite 持久化 Add/Search 服务、用户隔离、幂等写入、认证及媒体限制。
4. 可选重排专用代理，不要求将其他上游 API 全局代理。
5. 小样例测试、可续跑的全量图片预处理、部署配置与接口说明。

## 安装和启动

API 服务需要 Linux、Python 3.10+，可用 CPU 运行接口逻辑；模型推理由外部 API 提供。
2 核 4GB 的 CPU 实例不是本地大模型推理服务器。

```bash
cd methods/EverCore
python3 -m venv .venv
.venv/bin/pip install -r requirements-aml.txt
cp .env.aml.example .env.aml
chmod 600 .env.aml
```

在私有 `.env.aml` 中填写三个上游模型的 URL、Key，以及自行生成的随机服务访问令牌
`AML_API_TOKEN`（至少 32 字符）。该令牌不是模型厂商 Key。
配置不要提交 Git，也不要写入命令行参数或报告。

```bash
.venv/bin/python -m evaluation.src.aml serve \
  --env-file .env.aml --data-dir /absolute/path/to/private-memory \
  --host 127.0.0.1 --port 8080
```

生产部署模板见 `methods/EverCore/deployment/rawrag-aml.service`。
公网 IP 入口模板为 `nginx-ip-http.conf` / `nginx-ip-https.conf`，其中
`__PUBLIC_IP__` 需替换成自己的地址。HTTPS 模板需要先取得可验证的 IP 证书，
配置自动续期并放行必要安全组端口，不能直接使用缺少证书的模板启动。
公网开放还应遵守接入商及所在地适用要求，购买服务器不等于自动完成备案。

## API

- 无认证健康检查：`GET /health`。
- 写入：`POST /v1/memories/add`。
- 检索：`POST /v1/memories/search`。
- 认证支持 `Authorization: Token <AML_API_TOKEN>`、Bearer 或 `X-Api-Key`。

Add 示例：

```json
{
  "request_id": "example-write-1",
  "user_id": "example-user",
  "session_id": "example-session",
  "messages": [{"role": "user", "content": "I bought a red card in Paris."}]
}
```

写入落盘且可检索后，返回包含原始三个标识及 `success: true` 的 HTTP 200。
相同用户下相同 request_id 和相同载荷幂等；不同载荷明确冲突。

Search 示例：

```json
{"user_id": "example-user", "query": "Where did I buy the card?", "top_k": 100}
```

返回 `{"data": [{"id": "stable-id", "content": "source evidence", "score": 0.9}]}`。
实际返回不超过 `min(top_k, 15)`；无结果为 `{"data": []}`。
多模态 content/query 使用有序 `text` / `image_url` 数组，图片必须为内联 Base64
Data URI，不能传外部图片链接。完整核对表见 `evaluation/docs/AML_CONTRACT_ZH.md`。

## 运行限制与数据治理

- JPEG/PNG/WebP；单张图片解码不超过 10 MiB。
- 单次 Add 媒体总量及单次 Search 响应媒体总量各不超过 30 MiB。
- 请求体最大 128 MiB；默认同时处理 2 个请求，超出容量返回 429。
- 一个数据目录只允许一个服务进程；不是分布式高可用服务。
- 缓存和记忆均按 user_id 隔离，session_id 不限制同用户跨会话检索。
- 不记录原始 API 请求正文。运营方应按赛事规则安排清理，不允许训练或另作分发。
- 停止服务后可用 `python -m evaluation.src.aml purge-user --env-file .env.aml --data-dir <目录> --user-id <用户>` 删除该用户数据。
- 不保证未压测的大规模容量；临时电脑代理和手工 SSH 通道不等于无人值守生产出口。

## MemGallery 复现

先安装离线评测依赖 `requirements-multimodal.txt`，数据按原数据集条款单独获取。
默认不使用数据集现成图片说明，也不把答案或 clue 输入检索。

```bash
python -m evaluation.src.multimodal.full_run --stage captions \
  --data-root /absolute/path/to/Mem-Gallery --env-file .env.aml \
  --output /absolute/path/to/run --caption-workers 2
```

该步骤处理完整 20 场景的 1,003 张历史图片和 487 张查询图片，缓存可续跑。
Cohere Trial 每月 1,000 次调用不足以重排全部 1,711 道题；确认有足够合法额度后：

```bash
python -m evaluation.src.multimodal.full_run --stage evaluate \
  --data-root /absolute/path/to/Mem-Gallery --env-file .env.aml \
  --output /absolute/path/to/run --confirm-full-rerank-quota
```

不要在没有额度时使用确认标志，也不要通过轮换账号绕过限制。
Reader 每题最多 10 张图片，优先保留全部查询图；其余历史图预算及截断计数写入结果。
离线读取图片的体积限制与比赛 API 不同：研究流程最多 20 MiB，API 最多 10 MiB。
EM/F1 和来源轮次召回仅为诊断指标，不是官方 LLM 评审结果。

## 已有验证

完整写入一个 MemGallery 场景的 406 条消息、30 张历史图片后，测试 3 道题
（含图片查询），来源证据召回分别为 100%、100%、50%。这不代表全部答对或全集成绩。
同一快照在云端检索复测一致；全新图片 Add/Search、未认证拒绝、用户隔离、
服务重启后持久化检查通过。全量结果尚未完成，不将运行中任务写成完成成绩。

## 测试

```bash
pip install pytest pytest-asyncio
python -m pytest tests/test_aml_api.py tests/test_aml_smoke.py -q
```

安装离线评测依赖后，可进一步运行 `tests/test_memgallery_*.py` 和
`tests/test_multimodal_media.py`。测试使用合成凭据，不需要真实模型 Key。

## 参考

- 上游代码与论文：https://github.com/Litmeb/RAW-RAG
- AML 接口契约：https://agentmemoryleaderboard.ai/api-guide
- Cohere 调用限制：https://docs.cohere.com/docs/rate-limits
- MemGallery：https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery
