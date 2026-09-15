# 架构与设计取舍

这份文档解释**为什么这么写**，以及要改东西时应该动哪里。

## 一、分层的原则

```
入口层      app.py (Streamlit)   rag.py (CLI)   api.py (FastAPI)
              └──────────────┬──────────────┘
编排层              ingest.py        query.py        evaluate.py
能力层      loaders / headings / chunking / retrieval / llm
基础层          config.py   docmodel.py   vectorstore.py   bm25.py
```

两条硬规则：

1. **入口层不许有业务逻辑**。三个入口共用同一套能力层，所以界面能做的命令行和
   API 也能做，改一处三处都变。想验证某个功能有没有被硬编码进界面，看别的入口能不能用。
2. **只有 `vectorstore.py` 直接碰向量库和 embedding 模型**。换模型、换库、加缓存
   都只改一个文件。

## 二、数据模型：为什么先变成 Block 再变成 Section

```python
Block(text, page, locator)     # 一行文本 + 它从哪来
Section(path, blocks)          # 章节路径 + 属于它的行
```

如果把"读文件"和"切章节"揉在一起，每加一种格式就要重写一遍标题识别逻辑。
拆开之后：

- `loaders.py` 只回答「这个文件有哪些行、每行在第几页」；
- `headings.py` 只回答「这些行怎么组成章节」。

于是 **docx 和 PDF 共用同一套章节识别**，能力不会因为换了格式就下降——只是
"行从哪里来"不同。加一种新格式（比如 `.epub`）只要写一个 30 行的 loader，
不用碰标题逻辑。

## 三、分块：两个容易被忽略的细节

**1）章节前缀**

```
[第八九回 黄狮精虚设钉钯宴 金木土计闹豹头山] 又见那铁匠人等造成了三般兵器……
```

中文长文档里，一个 500 字的块常常看不出它在讲谁（"他道：'……'"）。
把章节路径拼进向量化的文本里，召回明显变好。展示时用
`chunking.strip_context_prefix()` 剥掉，避免和界面上的章节标题重复。

**2）稳定块 ID**

```python
chunk_id = sha1(source + section + section_index + text)[:24]
```

内容不变 → ID 不变 → 用 `upsert` 重复入库是**幂等**的，不会越建越多重复块。
配合「按 `source` 元数据删旧块」，就能做到**只重跑改过的那个文件**，
而不是每次都把整库推倒重建。

## 四、检索：三个必须处理的边界

融合部分看 README，这里只记实现时踩到的坑：

| 问题 | 处理 |
| --- | --- |
| 关键词路召回的块没有向量，MMR 算不了相似度 | `_ensure_embeddings()` 用 `collection.get(ids=...)` 一次性补齐 |
| 关键词命中的块 `similarity is None`，阈值过滤会误杀 | 过滤条件写成「相似度为空**或**高于阈值」 |
| 每次查询重建 BM25 索引 | 用 `(集合名, 块数, 总字符数)` 当指纹缓存，重建过库才失效 |
| BM25 分数与向量相似度量纲不同 | 用 RRF 按**排名**融合，不做数值相加 |
| 短查询里虚词抢权重 | 查询侧停用词过滤（文档侧不过滤，否则改变语料分布） |

另外，`_vector_candidates()` 一次 `collection.query()` 同时取回
`documents / metadatas / distances / embeddings`，避免为了拿向量再查一遍。

## 五、可测试性

`vectorstore.create_embeddings()` 是唯一的模型入口，测试里直接
monkeypatch 成确定性的假 embedding，于是：

- 测试不需要 GPU、不需要网络、不受模型版本波动影响；
- 端到端链路（入库 → 检索 → 生成）几秒就能验证；
- 反过来还验证了「重依赖是否真被懒加载」——CI 里压根没装 torch。

`tests/conftest.py` 里的 `isolated_store` 装置把 `CHROMA_DIR` 指到临时目录，
测试之间互不污染，也不会碰到你本机的向量库。

## 六、扩展点

| 想做的事 | 动哪里 |
| --- | --- |
| 支持新格式（epub、pptx…） | `loaders.py`：写 loader + 注册到 `SUPPORTED_EXTENSIONS` 和 `_LOADERS` |
| 支持新文体标题（病历、招股书…） | `headings.py`：往 `_FIXED_HEADINGS` 加词，或加一条正则 |
| 换向量模型 | `.env` 改 `EMBEDDING_MODEL`，然后 **重新建库** |
| 加一路检索（标题检索、时间过滤…） | `retrieval.py`：加一个 `_xxx_candidates()`，在 `_fuse()` 里加一路权重 |
| 接入新大模型 | `llm.py`：`PROVIDER_DEFAULTS` 加一条；OpenAI 兼容的直接复用 |
| 改提示词 | `query.py` 的 `SYSTEM_PROMPT` |
| 换向量库（Qdrant、Milvus…） | `vectorstore.py` 里的 collection 相关函数 |

## 七、已知取舍

- **BM25 用纯 Python 实现**。几万块以内够快（本仓库 2064 块，单次 60 ms 上下），
  上百万块应该换 `bm25s` 或 Elasticsearch。
- **MMR 默认关闭**。它是"覆盖度"工具，不是"精确度"工具，实测会拉低单点事实查询。
- **没有做查询改写（HyDE / 多查询）**。会显著增加延迟与 token 成本，留给需要的人自己加。
- **没有真正的多用户隔离**。`API_TOKEN` 是单一共享密钥，够内网用，不够 SaaS。
