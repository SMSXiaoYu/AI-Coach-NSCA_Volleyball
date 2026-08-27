# 🏐 AI Coach NSCA Volleyball

基于 RAG（检索增强生成）的排球运动知识检索系统。支持从多本中英文书籍中检索知识，并通过本地大模型生成中文回答。

## 🛠 技术栈

Python + Chroma + Milvus + SentenceTransformer + Cross-Encoder + Ollama(Qwen2.5)

> - **Phase 1**：Chroma（MVP）
> - **Phase 2**：Milvus（分布式索引）
> - **Phase 3**：HyDE + Rerank（检索优化）

## 📁 项目结构

```
src/
├── preprocess.py            # 提取书籍文字
├── ingest.py                # 构建向量库（Chroma）
├── retrieval.py             # 纯检索（Chroma 版）
├── generation.py             # 检索 + 生成（Chroma 版主程序）
├── retrieval_milvus.py      # 纯检索（Milvus 版）
├── generation_milvus.py     # 检索 + 生成（Milvus 版主程序）
├── retrieval_hyde_rerank.py # 纯检索（HyDE + Rerank）
├── generation_hyde_rerank.py # 检索 + 生成（HyDE + Rerank，当前主程序）
└── migrate_to_milvus.py     # 数据迁移脚本（Chroma → Milvus）
```

## 🚀 快速开始

### Phase 1（Chroma 版）

```bash
# 安装依赖
pip install -r requirements.txt
# 构建向量库（首次运行或新增书籍时执行）
python src/ingest.py
# 启动 AI 教练
python src/generation.py
```

### Phase 2（Milvus 版）

```bash
# 安装依赖
pip install -r requirements.txt
# 启动 Docker 并拉起 Milvus
cd D:\Milvus
docker-compose up -d
# 数据迁移（首次从 Chroma 迁移到 Milvus 时执行）
python src/migrate_to_milvus.py
# 启动 AI 教练（Milvus 版）
python src/generation_milvus.py
```

### Phase 3（HyDE + Rerank 版）

```bash
# 安装依赖
pip install -r requirements.txt
# 启动 Docker 并拉起 Milvus
cd D:\Milvus
docker-compose up -d
# 启动 AI 教练（HyDE + Rerank 版）
python src/generation_hyde_rerank.py
```

## 📌 当前状态

- ✅ Phase 1 已完成（2026-08-25）：4 本书，10609 个向量，支持中英文跨语言检索
- ✅ Phase 2 已完成（2026-08-27）：Chroma → Milvus 升级（分布式索引）
- ✅ Phase 3 已完成（2026-08-27）：HyDE + Rerank 检索优化
- 🔜 Phase 4-5 规划中

> 最后更新：2026-08-27

## 📄 License

[MIT](./LICENSE)
