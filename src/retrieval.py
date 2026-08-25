import os
import chromadb

# ==================== 环境配置 ====================
# 如果你之前下载 ONNX 时卡住，取消下面这行的注释（等下载完再注释掉）
# os.environ["CHROMA_ONNX_DISABLE"] = "true"

# ==================== 加载向量库 ====================
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("ai_coach")

# ==================== 自动获取知识库信息 ====================
def get_kb_info(collection):
    """自动读取向量库中的书籍数量和名称"""
    try:
        # 获取所有元数据
        result = collection.get()
        metadatas = result.get('metadatas', [])
        
        if not metadatas:
            return "空知识库", 0, []
        
        # 提取所有不同的来源（书籍文件名）
        sources = list(set([
            m.get('source', '未知文件') 
            for m in metadatas 
            if m and m.get('source')
        ]))
        
        total_chunks = len(metadatas)
        
        # 格式化书籍名称显示
        if len(sources) <= 3:
            books_display = ", ".join(sources)
        else:
            books_display = f"{len(sources)} 本书（{', '.join(sources[:3])} 等）"
        
        return books_display, total_chunks, sources
        
    except Exception as e:
        return f"读取失败: {e}", 0, []

# ==================== 启动界面 ====================
books_display, total_chunks, sources = get_kb_info(collection)

print("=" * 50)
print("🏀 AI 教练知识库 - 问答模式")
print(f"📚 当前知识库: {books_display}")
print(f"📊 向量总数: {total_chunks} 个")
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
    
    # 检索
    results = collection.query(query_texts=[query], n_results=3)
    
    print("\n📖 检索结果:\n")
    
    if results and results.get('documents') and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0], 1):
            source = results['metadatas'][0][i-1].get('source', '未知')
            print(f"【结果 {i}】来源: {source}")
            print(f"{doc[:300]}...")
            print("-" * 40)
    else:
        print("❌ 未找到相关内容")
    
    print()