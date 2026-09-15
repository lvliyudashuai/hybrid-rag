# 通用 RAG 文档问答系统

[English](README.en.md) | 简体中文

> 把一堆文档（docx / pdf / txt / md / html / csv / xlsx / json）丢进去，然后像聊天一样提问，
> 得到**带出处引用**的答案。全流程可离线跑通，不填 API Key 也能用（走离线模拟）。

这是一个**可读、可改、可评估**的 RAG 工程实现，不是玩具 demo：
每个环节（加载 / 分块 / 检索 / 生成 / 评估 / 服务化）都拆成独立模块并有测试覆盖。

![界面预览](docs/screenshot.png)

不填 API Key 也能用：检索照常工作，回答退化成离线拼装，全程不碰任何外部服务。

![离线模式（无 API Key）](docs/screenshot-offline.png)

---

## 它比"能跑"多了什么

| 能力 | 说明 |
| --- | --- |
| **多格式 + 批量入库** | 8 种格式，目录递归扫描，一次可入库多个文档 |
| **结构感知分块** | 识别「第X回 / 第X章 / 1.2.3 / # Markdown / 参考文献」等标题，按章节切分，块里带上章节路径 |
| **混合检索** | 向量召回 + BM25 关键词召回，用 RRF 融合（不是简单加权，原因见下方） |
| **可选出分** | MMR 去重、相似度阈值、CrossEncoder 重排，全部开关化 |
| **引用溯源** | 答案带 `[1][2]` 编号，界面上能对回原文的章节和页码 |
| **增量更新** | 块 ID 由内容哈希决定，同一文档重复入库不会产生重复块；改一个文件只重跑那一个 |
| **离线评估** | 自带评估集与指标（Hit@k / MRR / 关键词覆盖），改参数前后能对比数字 |
| **可观测** | 检索链路每一步的耗时与候选数都有 trace；统一日志、环境自检 |
| **三种入口** | Streamlit 界面 / 命令行 / REST API，共用同一套核心代码 |
| **测试与 CI** | 74 个测试，不依赖网络与真实模型，GitHub Actions 每次提交都跑 |

---

## 快速开始

```bash
git clone https://github.com/lvliyudashuai/hybrid-rag.git
cd hybrid-rag

python -m venv .venv
.venv\Scripts\activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # 按需修改；不填 Key 也能跑
python rag.py --doctor          # 环境自检：模型、显卡、向量库、Key 一眼看清
python rag.py --rebuild         # 建库（首次会下载 embedding 模型）
python rag.py "这份文档讲了什么？"   # 命令行问答
streamlit run app.py            # 打开界面
```

GPU 用户想装 CUDA 版 torch，请按 [pytorch.org](https://pytorch.org) 的指引先装 torch 再装其余依赖。
不填任何 API Key 时，系统走**离线模拟**：检索照常工作，回答由最相关的片段拼成，不调用任何外部服务。

### Windows：双击就能用

装完依赖后直接双击 `start.bat`：它会先看服务是不是已经在跑，没有就起一个，
等健康检查通过再打开「应用窗口」——所以不会开出一个白屏。**关掉那个控制台窗口就等于停止服务。**

想给它在桌面放个图标（图标借用 Python 的，按需改路径）：

```powershell
$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = $shell.CreateShortcut("$desktop\RAG 文档问答助手.lnk")
$lnk.TargetPath = "$PWD\start.bat"
$lnk.WorkingDirectory = "$PWD"
$lnk.Save()
```

项目自带的 `.streamlit/config.toml` 打开了 `server.runOnSave`：改完 `.py` 保存，页面自动重载，不用手动点 Rerun。

---

## 目录结构

```
config.py         所有配置的唯一来源（.env 覆盖 + 运行时覆盖 + 类型校验）
docmodel.py       Block / Section 两个数据模型
loaders.py        8 种格式的加载器 + 目录/通配符展开
headings.py       标题识别引擎（docx/pdf/文本共用一套）
chunking.py       分块 + 章节前缀 + 稳定块 ID
vectorstore.py    embedding 模型与 Chroma 存取、统计、残留清理
bm25.py           BM25 关键词检索（jieba 分词 + 查询侧停用词）
retrieval.py      混合检索：双路召回 → RRF 融合 → MMR → 重排
llm.py            大模型调用（DeepSeek/OpenAI/Claude/Ollama，流式、重试）
query.py          提示词构造、引用、回答生成
ingest.py         入库编排（批量、增量、失败隔离）
evaluate.py       离线评估（Hit@k / MRR / 关键词覆盖）
app.py            Streamlit 界面        rag.py 命令行        api.py REST 接口
launch.py         一键启动器（起服务 → 等就绪 → 开窗口）    start.bat 双击入口
.streamlit/       Streamlit 本地配置（保存自动重载、关闭统计上报）
tests/            74 个测试（用假 embedding，不联网）
eval/             评估集与评估结果
```

数据流：

```
文档 → loaders → headings → chunking → vectorstore(Chroma)
                                          │
   问题 ──┬───────────────────────────────┴──→ retrieval(混合检索) → query(生成) → 带引用的答案
          └──→ bm25
```

---

## 检索：为什么要混合，怎么混合

**单靠向量**：问法灵活，但专有名词、编号、型号、法条号这类"精确词"经常漏（小模型几乎没学到它们的语义）。
**单靠 BM25**：精确词命中准，但换个说法就失效（"退货流程" 匹配不到 "退款步骤"）。

所以两路都召回，再融合。**融合用 RRF（按排名）而不是加权分数**，因为向量相似度在 0~1 之间，
BM25 分数可以是 0~20+，直接相加会被 BM25 单方面主导：

```
score = α · 1/(k + 向量排名) + (1-α) · 1/(k + 关键词排名)      # k 默认 60
```

几个容易踩的坑，这里都处理了：

- 关键词路召回的块没有向量，做 MMR 前要**补取向量**；
- 只用关键词命中的块没有相似度，阈值过滤时不能误杀；
- 每次查询重建 BM25 索引会很慢，用「集合名 + 块数 + 总字符数」做指纹缓存。

---

## 评估：改参数不看数字就是瞎调

```bash
python evaluate.py                      # 三种模式对比
python evaluate.py --show-failures      # 看哪些问题没召回对
python evaluate.py --json eval/last_report.json
```

评估集是 JSONL，一条一个问题：

```json
{"question": "猪八戒的武器有多重？", "expected_sections": ["第八九回"],
 "expected_keywords": ["八百斤"], "answerable": true}
```

**本仓库实测**（示例文档 90 回 / 2064 块，14 个问题，k=5，Qwen3-Embedding-0.6B，RTX 5070）：

| 模式 | Hit@5 | MRR | 关键词覆盖 | 平均耗时 |
| --- | --- | --- | --- | --- |
| 纯向量 | 78.6% | 0.386 | 64.3% | 59 ms |
| 纯关键词 BM25 | 42.9% | 0.157 | 50.0% | 63 ms |
| **混合（α=0.9）** | **78.6%** | **0.410** | **64.3%** | 159 ms |

几点如实说明：

- 这份文档是古白话，BM25 明显吃亏，所以最优 α 偏高（0.9 时混合≈向量+关键词兜底）。
  如果你的文档里充满编号、法条、型号，应把 `HYBRID_ALPHA` 调到 0.5~0.7，**用你的数据跑一遍评估再定**。
- `USE_MMR` 默认**关闭**：实测在"查一个具体事实"的场景下 MMR 会把正确答案挤下去（78.6% → 71.4%）。
  问"文档都讲了哪些方面"这类需要覆盖度的问题时再打开。
- 14 个问题的样本很小，一个问题的差异就是 7%，别过度解读单次波动。

---

## 配置

全部配置项都在 `.env`，也可以直接在界面「设置」页改。常用几个：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 中文效果好；省空间可换 `BAAI/bge-small-zh-v1.5` |
| `EMBEDDING_DEVICE` | `auto` | 自动选 cuda / mps / cpu |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | 中文文档 300~800 一般合适 |
| `CHUNK_CONTEXT_PREFIX` | `1` | 块前拼章节路径，长文档召回明显更好 |
| `RETRIEVAL_MODE` | `hybrid` | `vector` / `keyword` / `hybrid` |
| `HYBRID_ALPHA` | `0.9` | 向量路权重，精确词多时调低 |
| `USE_RERANK` | `0` | 打开会下载 CrossEncoder 并做精排 |
| `MAX_CONTEXT_CHARS` | `6000` | 拼给模型的资料上限，防止烧 token |
| `DEEPSEEK_API_KEY` 等 | 空 | 留空走离线模拟；界面上临时填的 Key 只在内存里 |

---

## 三种用法

**命令行**

```bash
python rag.py --doctor                    # 环境自检
python rag.py --rebuild                   # 重建索引
python rag.py --ingest ./docs --append    # 追加目录（增量，不动已有文档）
python rag.py "问题" --mode hybrid --trace # 单次问答 + 链路耗时
python rag.py                             # 交互式（支持多轮上下文）
python rag.py --stats / --prune           # 库状态 / 清理残留目录
```

**Streamlit 界面**：`streamlit run app.py`，或直接双击 `start.bat`（等价于 `python launch.py`）
对话（流式输出 + 引用，可删单条或清空整个对话）/ 知识库（上传、增量入库、清空）/ 检索调试（三种模式并排对比）/ 设置 / 关于。

**REST API**：`python api.py`，文档在 `http://127.0.0.1:8000/docs`

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "合同里的违约责任怎么约定的？", "k": 5}'
```

设了 `API_TOKEN` 后所有 `/v1` 接口都要带 `X-API-Key`。不设就是开放访问，**别直接暴露到公网**。

---

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests -q      # 74 个测试，约 25 秒，不需要网络和真实模型
ruff check .
```

测试用确定性的假 embedding 顶替真实模型：CI 里几秒出结果，也不受模型版本波动影响；
同时这反过来验证了「重依赖是否真被懒加载」——只装核心依赖也能跑测试。

---

## 常见问题

**Q：第一次提问很慢？**
首次调用要把 embedding 模型加载进显存（0.6B 模型约 5~20 秒），之后就快了。
`--trace` 会把「向量召回 / BM25 / 融合」各自的耗时打出来，一眼看出慢在哪。

**Q：扫描版 PDF 读不出内容？**
图片型 PDF 没有文字层，`pypdf` 提不出文本。程序会明确报错而不是悄悄建一个空库。需要先做 OCR。

**Q：向量库越用越大？**
Chroma 删除/重建 collection 会留下旧的 HNSW 段目录。`python rag.py --prune` 只删
"形如 UUID 且没有登记在 sqlite 里"的目录，不会误伤数据。

**Q：我的 Key 会不会被提交上去？**
`.env` 在 `.gitignore` 里；界面上临时输入的 Key 只存在内存中，只有你显式勾选
「把 API Key 也写入 .env」才会落盘；日志和报错信息里的 Key 一律脱敏成 `sk-1****abcd`。

**Q：回答是"资料中未找到相关内容"？**
说明检索到的片段里确实没有答案——这是设计行为（不编造）。可以试试调大 `TOP_K`、
把 `RETRIEVAL_MODE` 换成 `hybrid`、或者确认文档真的入库了。

---

## 后续可做

- [ ] 重排模型实测对比（`USE_RERANK=1` 的效果数据）
- [ ] 父子块检索（小块检索、大块给模型）
- [ ] OCR 接入，支持扫描版 PDF
- [ ] 答案级评估（忠实度 / 拒答正确率），需要真实模型参与
- [ ] Docker 化与多用户鉴权

## License

[MIT](LICENSE)
