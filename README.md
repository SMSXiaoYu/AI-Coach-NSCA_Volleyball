# 🏐 AI Coach NSCA Volleyball

基于 RAG（检索增强生成）的排球运动知识检索系统。支持从多本中英文书籍中检索知识，并通过本地大模型生成中文回答。

## 🛠 技术栈

Python + Chroma + SentenceTransformer + Ollama(Qwen2.5)

## 📁 项目结构

```
src/
├── preprocess.py    # 提取书籍文字
├── ingest.py        # 构建向量库
├── retrieval.py     # 纯检索
└── generation.py     # 检索 + 生成（主程序）
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
# 构建向量库（首次运行或新增书籍时执行）
python src/ingest.py
# 启动 AI 教练
python src/generation.py
```

## 📌 当前状态

- ✅ Phase 1 已完成：4 本书，10609 个向量，支持中英文跨语言检索
- 🔜 Phase 2-5 规划中

## 📄 License

[MIT](./LICENSE)
