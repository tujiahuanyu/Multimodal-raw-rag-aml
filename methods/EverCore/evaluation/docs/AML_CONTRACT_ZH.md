# AML 接口核对与小规模测试

核对日期：2026-10-26。
官方依据：https://agentmemoryleaderboard.ai/api-guide 。
本服务实现同步 Add/Search，Search 不生成最终答案；Answer/Eval 由平台完成。
本轮实际结果与未完成项见 [2026-10-26 复查记录](AML_AUDIT_20261026_ZH.md)。

版本说明：本页历史公网核验属于此前部署版本。当前 v1.3 开发代码在传入的历史
caption 之外新增图片多描述索引；接口契约不变，但索引身份已改变，必须使用新的数据目录；
更新服务时保留旧版数据并使用全新索引，线上版本以 /health 为准，详见 [v1.3 说明](MULTI_VIEW_IMAGE_RETRIEVAL_ZH.md)。

| 官方要求 | 服务实现 |
| --- | --- |
| Add 接收 request_id、user_id、session_id、原始顺序的 messages | POST /v1/memories/add，保留原始标识和消息顺序 |
| messages 的角色为 user/assistant，时间可选且为 Unix 毫秒 | 严格校验字段类型和时间 |
| Add 成功前持久化且立即可检索 | SQLite 事务写入后才返回 HTTP 200，并原样返回三个标识及 success=true |
| 重试使用相同 request_id | 相同用户、相同请求号和相同载荷幂等，不重复模型调用；不同载荷明确冲突 |
| Search 包含原始 query、user_id、top_k，选择题 options 单独放顶层 | POST /v1/memories/search，支持文字和有序图文 query |
| 按 user_id 隔离，不能跨样本共享状态 | 记忆和派生缓存均按用户隔离；session_id 用于组织，不限制跨会话检索 |
| 响应为 data 数组，条目有稳定 id、非空 content，按相关性排序 | 返回原始来源证据，包含可选 score 和来源时间 |
| 数量不得超过 top_k，正式评测 top_k=100 | 一次重排全部融合候选，按顺序展开并按原始消息 ID 去重，再选最多 min(top_k, 15) 条完整消息；空结果为 data=[] |
| 多模态保持 text/image_url 顺序，只接受内联 Base64 | 存储并返回原始图片，不以图片网址代替，不静默缩放 |
| JPEG/PNG/WebP，单图片解码不超过 10 MiB | 校验实际格式、MIME 和体积，明确拒绝不支持的输入 |
| Add 总媒体不超过 30 MiB，Search 响应总媒体不超过 30 MiB | 校验 Add；Search 按预算选择完整证据条目，不裁剪单个证据 |
| 接口使用认证，健康检查无需认证 | 支持 Authorization: Token/Bearer 和 X-Api-Key；GET /health 无需认证 |
| 单次最长 30 分钟 | 服务同步处理，部署时的入口代理也必须允许足够的请求时间 |

云端已配置可信公网 IP HTTPS 入口、安全组及证书自动续期；内部 API 和重排代理
仍仅回环监听。不要通过公网明文 HTTP 传输真实 Key 或评测记忆。

## 官方样例回归

`evaluation/config/aml_official_examples.json` 记录官方契约正文和接入指南中的两组文字
Add/Search 样例，以及一组多模态 Add/Search 样例。保留原始字段、标识和 options。
官方图片 Base64 使用不可执行的省略占位符，测试仅将这些占位符替换为生成的有效
JPEG/PNG，不能声称没有任何替换就直接发送了原始图片字符串。

本机接口回归使用替身模型，不产生真实 API 费用：

```bash
python -m pytest tests/test_aml_api.py tests/test_aml_smoke.py tests/test_aml_contract_validation.py -q
```

对已经部署的真实 HTTPS 服务进行样例自测：

```bash
python -m evaluation.src.aml.contract_smoke \
  --base-url https://YOUR_PUBLIC_ENDPOINT \
  --env-file .env.aml --output /private/path/official-example-report.json
```

该命令会产生少量真实图片描述、向量和重排调用。它检查响应结构、标识回显、立即可检索、
幂等重放、冲突、来源内容、图文顺序、稳定 ID、Top K、三种鉴权、空用户和非法输入。
报告不保存 Key、请求正文或返回证据。默认检索间隔 6.2 秒，不改变服务器并发配置。
这是参赛方运行的“官方文档样例自测”，不是主办方签发的正式 Smoke 或 Answer/Eval。

本轮修复：即使 user_id 尚无记忆或 top_k=0，也先校验查询内容，非法图片和空白查询
返回 422，而不是悄悄返回成功空结果。来源与方法改动见 [披露说明](METHOD_DISCLOSURE_ZH.md)。
2026-10-27 的 v1.1 源码新增一次重排后的完整消息去重及后置限额；旧核查报告记录的是此前版本，
不能将旧公网测试或平台 v1.0 Smoke 成绩作为新流程的测试结果。

## 仍需完成的运营与平台条件

- 云端 Cohere 仍依赖电脑代理；自动重连不等于独立、无人值守的网络出口。
- Trial 调用额度及完整赛道的存储、并发、流量预算仍需保障，未完成长期容量压测。
- 需取得平台 Eval Key 后运行正式 Smoke；自测不能替代主办方校验。
- 正式任务结束后 30 天内，必须清理记忆、派生缓存和所有备份副本。当前提供停服后的
  `purge-user` 命令，但没有平台任务结束回调或自动到期清理，不得把它描述成已自动完成。
- 需核对上游模型服务提供方的数据处理条款；本轮代码检查只确认调用配置和接口行为，
  不能独立证明第三方服务实际模型权重或数据留存行为。

## MemGallery 检查范围

新增命令 `python -m evaluation.src.aml.memgallery_smoke prepare/run`。
使用一个完整场景历史和数据集提供的历史 caption，按原始会话顺序分批 Add；
默认测试三个问题，按无图／有图交替选择，可通过 --questions 调整数量。
不把查询图片的现成 caption 放入请求，Search 从查询图片生成说明。
原始答案和 clue 不进入 Add 或 Search；clue 只在检索完成后计算诊断召回率。
图片超过官方限制时明确报错，不默默缩放或跳过。
测试数据包和参考标注分开保存，不让服务读取参考标注。
该小样例不是官方 Answer/Eval，也不是 MemGallery 全集成绩。
