# AML 接口核对与小规模测试

核对日期：2026-10-26。
官方依据：https://agentmemoryleaderboard.ai/api-guide 。
本服务实现同步 Add/Search，Search 不生成最终答案；Answer/Eval 由平台完成。

| 官方要求 | 服务实现 |
| --- | --- |
| Add 接收 request_id、user_id、session_id、原始顺序的 messages | POST /v1/memories/add，保留原始标识和消息顺序 |
| messages 的角色为 user/assistant，时间可选且为 Unix 毫秒 | 严格校验字段类型和时间 |
| Add 成功前持久化且立即可检索 | SQLite 事务写入后才返回 HTTP 200，并原样返回三个标识及 success=true |
| 重试使用相同 request_id | 相同用户、相同请求号和相同载荷幂等，不重复模型调用；不同载荷明确冲突 |
| Search 包含原始 query、user_id、top_k，选择题 options 单独放顶层 | POST /v1/memories/search，支持文字和有序图文 query |
| 按 user_id 隔离，不能跨样本共享状态 | 记忆和派生缓存均按用户隔离；session_id 用于组织，不限制跨会话检索 |
| 响应为 data 数组，条目有稳定 id、非空 content，按相关性排序 | 返回原始来源证据，包含可选 score 和来源时间 |
| 数量不得超过 top_k，正式评测 top_k=100 | 固定最多重排返回 15 条，同时不超过调用方 top_k；空结果为 data=[] |
| 多模态保持 text/image_url 顺序，只接受内联 Base64 | 存储并返回原始图片，不以图片网址代替，不静默缩放 |
| JPEG/PNG/WebP，单图片解码不超过 10 MiB | 校验实际格式、MIME 和体积，明确拒绝不支持的输入 |
| Add 总媒体不超过 30 MiB，Search 响应总媒体不超过 30 MiB | 校验 Add；Search 按预算选择完整证据条目，不裁剪单个证据 |
| 接口使用认证，健康检查无需认证 | 支持 Authorization: Token/Bearer 和 X-Api-Key；GET /health 无需认证 |
| 单次最长 30 分钟 | 服务同步处理，部署时的入口代理也必须允许足够的请求时间 |

正式参赛还要求可公网访问的接口、固定版本及公开仓库材料。
云端仅回环监听的测试服务不等于已具备公开提交条件，需再配置 HTTPS 入口及安全组。
不要通过公网明文 HTTP 传输真实 Key 或评测记忆。

## MemGallery 检查范围

新增命令 `python -m evaluation.src.aml.memgallery_smoke prepare/run`。
使用一个完整场景历史，按原始会话顺序分批 Add，再测试三个问题（包含图片查询）。
原始答案和 clue 不进入 Add 或 Search；clue 只在检索完成后计算诊断召回率。
图片超过官方限制时明确报错，不默默缩放或跳过。
测试数据包和参考标注分开保存，不让服务读取参考标注。
该小样例不是官方 Answer/Eval，也不是 MemGallery 全集成绩。
