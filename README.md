# RAG 文档问答 Demo

一个端到端的检索增强生成（RAG）演示项目：指定一篇本地文档（docx 或 pdf），系统自动完成 **文档解析 → 章节元数据注入 → 分块 → 向量化 → 相似度检索 → LLM 生成回答**，并提供可交互的 Streamlit 前端与命令行入口。

> 适合作为学习 RAG 完整流程、搭建个人知识库问答系统的入门示例。

## 功能特点

- **多格式文档**：支持 `.docx` / `.pdf`，按扩展名或 `.env` 的 `DOC_TYPE` 自动选择解析器。
- **章节元数据注入**：自动识别文档的标题层级（如 `3.3 章节名`），把章节路径写入每个向量块，检索结果可溯源。
- **中文向量检索**：默认使用轻量的 `BAAI/bge-small-zh-v1.5`，开箱即用；可切换 `Qwen/Qwen3-Embedding-0.6B` 提升中文语义效果。
- **检索 + 生成闭环**：检索 Top-K 段落交给 DeepSeek / OpenAI / Claude 生成答案；未配置 API Key 时自动降级为离线模拟回答，便于演示。
- **相似度可视化**：把 Chroma 的余弦距离换算成直观的相似度百分比。
- **可交互前端**：Streamlit 页面支持调节 Top-K、温度、provider，并展开查看每个检索到的片段。
- **命令行入口**：支持一键建库、单次问答。

## 目录结构

```
rag_demo/
├── config.py             # 集中配置（可由 .env 覆盖）
├── ingest.py             # 文档解析 + 章节 + 分块 + 向量入库
├── vectorstore.py        # embedding 与 Chroma 封装（带模型缓存）
├── query.py              # 检索 + 生成
├── app.py                # Streamlit 前端
├── rag.py                # CLI 入口
├── requirements.txt
├── .env.example          # 配置模板
└── rag.docx              # 示例文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 国内网络较慢时建议使用镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
>
> 若无法直连 HuggingFace，`.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像下载模型。

### 2. 配置（可选）

项目自带 `.env`，默认使用本地缓存的中文 embedding 模型 + 离线回答，**无需任何 API Key 即可演示**。

想要真实生成回答，编辑 `.env`：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxxx
```

也可切换 OpenAI / Claude，参考 `.env.example`。`.env` 不会提交到 Git（已被 `.gitignore` 忽略），只有 `.env.example` 模板会上传。

### 3. 构建向量库

```bash
python ingest.py
```

首次运行会自动下载 embedding 模型。当前默认 `BAAI/bge-small-zh-v1.5`（约 192MB），向量库写入 `chroma_db_qwen/`。想换 Qwen3，在 `.env` 修改 `EMBEDDING_MODEL` 后重新建库。

### 4. 运行

Streamlit 交互界面：

```bash
streamlit run app.py
```

命令行：

```bash
python rag.py "这篇文章主要讲了什么？"
```

## 常见问题

- **检索不到内容？** 先执行 `python ingest.py` 确认建库成功。
- **切换向量模型？** 在 `.env` 修改 `EMBEDDING_MODEL`，然后重新执行 `python ingest.py`（维度不同需重建库）。
- **离线能跑吗？** 能。未填 Key 时使用 mock provider，会基于检索结果给出带来源的模拟回答。
- **换模型需要联网？** 首次下载需联网；已有缓存时建议在 `.env` 设 `EMBEDDING_LOCAL_ONLY=1` 加速并避免联网挂起。

## 待优化方向

- 使用 Qwen3-Embedding 时增加 query/passage 不对称检索指令。
- 接入混合检索（BM25 + 向量）与重排（Reranker）。
- 支持网页上传文档，做成通用文档问答应用。
