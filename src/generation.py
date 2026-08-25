import os
import chromadb
from openai import OpenAI

# ==================== 配置 ====================
# 本地 Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"  # 你下载的模型名称

# Chroma 向量库路径
CHROMA_DIR = "chroma_db"

# ==================== 加载向量库 ====================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("ai_coach")

# ==================== 初始化 LLM 客户端 ====================
llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # 本地服务不需要真实 key
)

# ==================== 知识库信息（复用 retrieval.py 的逻辑） ====================
def get_kb_info(collection):
    try:
        result = collection.get()
        metadatas = result.get('metadatas', [])
        if not metadatas:
            return "空知识库", 0
        sources = list(set([m.get('source', '未知文件') for m in metadatas if m and m.get('source')]))
        total_chunks = len(metadatas)
        if len(sources) <= 3:
            books_display = ", ".join(sources)
        else:
            books_display = f"{len(sources)} 本书（{', '.join(sources[:3])} 等）"
        return books_display, total_chunks
    except Exception as e:
        return f"读取失败: {e}", 0

# ==================== 检索函数 ====================
def retrieve(query, n_results=3):
    """从向量库检索相关文档"""
    results = collection.query(query_texts=[query], n_results=n_results)
    if results and results.get('documents') and results['documents'][0]:
        return results['documents'][0], results['metadatas'][0]
    return [], []

# ==================== 生成函数 ====================
def generate_response(query, context_docs, sources):
    """基于检索结果，调用 LLM 生成回答"""
    
    # 构建上下文
    context = "\n\n".join(context_docs)
    
    # 系统 Prompt
    system_prompt = """你是一名专业的排球体能教练。请根据以下书籍内容，用中文回答用户的问题。

要求：
1. 只基于提供的书籍内容回答，不要编造
2. 如果内容中没有相关信息，请明确说"书中未提及"
3. 回答要专业、清晰、有条理
4. 如果涉及训练建议，请给出具体的动作、组数、次数等细节"""

    user_prompt = f"""
书籍参考内容：
{context}

用户问题：{query}

请用中文回答："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM 调用失败: {e}"

# ==================== 启动界面 ====================
books_display, total_chunks = get_kb_info(collection)

print("=" * 50)
print("🏀 AI 教练 - 智能问答模式（检索 + 生成）")
print(f"📚 当前知识库: {books_display}")
print(f"📊 向量总数: {total_chunks} 个")
print(f"🤖 大模型: {OLLAMA_MODEL} (本地 Ollama)")
print("=" * 50)
print("💡 输入问题，按回车查询")
print("💡 输入 'exit' 或 'quit' 退出\n")

# ==================== 对话循环 ====================
while True:
    query = input("❓ 你的问题: ").strip()
    
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 再见！")
        break
    
    if not query:
        continue
    
    # 1. 检索
    print("\n🔍 正在检索...")
    docs, metadatas = retrieve(query, n_results=3)
    
    if not docs:
        print("❌ 未找到相关内容\n")
        continue
    
    # 2. 显示检索来源
    print("\n📖 参考来源:")
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        source = meta.get('source', '未知')
        print(f"   {i}. {source[:60]}...")
    
    # 3. 生成回答
    print("\n🤖 正在生成回答...")
    print("-" * 50)
    answer = generate_response(query, docs, metadatas)
    print(answer)
    print("-" * 50)
    print()