# 原始工作与方法改动披露

## 原始工作

- 论文：Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought。
- 原作者：Zhengye Bao、Shengtao Zhang、Haokai Zhang、Shiqi Lou、Muning Wen、Minnan Luo；姓名根据上游仓库 Citation 部分记录。
- 技术报告：https://arxiv.org/abs/2610.30755 。
- 上游代码：https://github.com/Litmeb/RAW-RAG 。
- 派生起点：`95639191e9547219b426710e50fe1a36235e26fc`，保留上游 MIT LICENSE。
- 本仓库为第三方多模态和比赛接口适配，不是原作者官方比赛系统，不将上游方法或论文成绩声明为适配者原创成果。

## 线上参赛服务的改动

1. 新增独立的同步 AML Add/Search 服务。保留“原始历史、向量召回、重排、返回证据”的思路，不直接使用离线评测 Reader 作为 Search。
2. 图片由 `gpt-4o-mini` 生成视觉描述，描述和原始会话文字参与文本向量检索；返回时保持历史文字、图片字节及其顺序。不使用原生图片向量检索或视觉重排。
3. 按来源 session 组织切片，按精确 user_id 检索隔离；添加 SQLite 持久化、事务写入、请求幂等、访问认证和用户内派生缓存。
4. 当前服务为 256-token 窗口、32-token 重叠，向量召回最多 128 个候选，Cohere 重排最多返回 15 条且不超过请求 top_k。
5. 返回候选窗口对应的完整来源消息，并附来源会话、角色和时间标签。多模态响应按媒体预算选择完整候选，不静默剪短单条来源。
6. 选择题 options 作为检索输入的附加上下文，保持官方 query 的原始文字；不接收答案或 clue，不生成最终答案。
7. 添加图像格式、字节和像素校验、请求容量限制、可重试错误响应、专用重排代理及 Linux/systemd/Nginx 部署配置。

这些服务化和多模态改动可能影响检索结果，不能声称是原论文全部细节不变的严格复现。

## 模型与实验边界

线上默认描述模型为 `gpt-4o-mini`，向量模型为 `text-embedding-v4`，重排为
`rerank-v3.5`。模型调用通过私有配置指定的服务提供方完成，仓库不包含真实凭据。

另有本地 GPU 的 `Qwen3-Reranker-4B` MemGallery 研究配置，与云端 Cohere 配置分开。
该流程使用公开数据进行图片描述、检索、Reader 及诊断指标评测；不是赛事平台
统一 Answer/Eval。不得将公开集研究分数或自测通过状态写成官方平台成绩。

运行细节见 [本地 Qwen 说明](MEMGALLERY_LOCAL_QWEN_ZH.md)，
接口和部署边界见 [AML 核对表](AML_CONTRACT_ZH.md)。
