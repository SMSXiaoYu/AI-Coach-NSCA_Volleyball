# 🏐 AI Coach NSCA Volleyball

基于 RAG（检索增强生成）的排球运动知识检索系统。支持从多本中英文书籍中检索知识，并通过本地大模型生成中文回答。

## 🛠 技术栈

Python + Chroma + Milvus + SentenceTransformer + Cross-Encoder + Ollama(Qwen2.5) + FastAPI + Gradio + Dify

> - **Phase 1**：Chroma（MVP）
> - **Phase 2**：Milvus（分布式索引）
> - **Phase 3**：HyDE + Rerank（检索优化）
> - **Phase 4**：FastAPI + Gradio（产品化）
> - **Phase 5**：Dify（低代码集成）

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
├── performance_validation.py # 性能验证（Phase 3.5 测试脚本）
├── phase5_dify_integration.py # Dify 集成示例（Phase 5）
└── migrate_to_milvus.py     # 数据迁移脚本（Chroma → Milvus）

docs/
├── phase35_results_*.txt    # Phase 3.5 实验报告
└── dify_demo.mp4            # Phase 5 演示视频

api/
└── main.py                  # FastAPI 后端服务

frontend/
└── app.py                   # Gradio 前端界面
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

### Phase 4（Web 产品版）

```bash
# 安装依赖
pip install -r requirements.txt
# 启动 Docker 并拉起 Milvus
cd D:\Milvus
docker-compose up -d
# 启动后端（终端1）
python api/main.py
# 启动前端（终端2）
python frontend/app.py
```

### Phase 5（Dify 低代码版）

```bash
# 1. 启动 Dify（首次需在 dify-main/docker/.env 中配置）
cd dify-main/docker
docker-compose -f docker-compose.yaml up -d
# 浏览器访问 http://localhost 完成 setup（创建管理员账户）

# 2. 在 Dify 界面：创建知识库 → 导入文档 → 创建 Chatflow 应用 → 获取 API Key

# 3. 运行集成验证
python src/phase5_dify_integration.py

# 架构：原有 FastAPI 代码壳不变，RAG 核心改为调用 Dify API
```

## 📌 当前状态

- ✅ Phase 1 已完成（2026-08-25）：4 本书，10609 个向量，支持中英文跨语言检索
- ✅ Phase 2 已完成（2026-08-27）：Chroma → Milvus 升级（分布式索引）
- ✅ Phase 3 已完成（2026-08-27）：HyDE + Rerank 检索优化
- ✅ Phase 3.5 已完成（2026-08-28）：Performance Validation，首条命中率从 27.3% 提升至 45.5%
- ✅ Phase 4 已完成（2026-08-30）：Productization，FastAPI + Gradio Web 应用
- ✅ Phase 5 已完成（2026-08-31）：Low-Code Integration，Dify 本地部署 + Chatflow 搭建

> 最后更新：2026-08-31

## 🎥 演示视频

[Phase 5：Dify 集成演示](./docs/dify_demo.mp4)

## 📄 License

[MIT](./LICENSE)
