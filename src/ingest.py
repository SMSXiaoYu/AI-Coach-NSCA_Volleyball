import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import shutil
from pathlib import Path
import chromadb

# ==================== 配置 ====================
PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


SELECTED_BOOKS = [
    "Essentials of Strength Training and Conditioning -- NSCA.txt",
    "The_Vertical_Jump_Development_Bible.txt",
    "Coaching Volleyball Successfully.txt",
    "体育健身训练丛书：营养 训练 放松 损伤预防（套装全10册）.txt",
]

# ==================== 自定义 PaddleNLP Embedding 函数 ====================
class PaddleNLPEmbeddingFunction:
    """使用 PaddleNLP 的 Ernie 模型进行 Embedding"""
    
    def __init__(self, model_name="rocketqa-zh-dureader-query-encoder"):
        print(f"   📂 正在加载 PaddleNLP 模型: {model_name}")
        print("   ⏳ 加载模型，请稍候...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        print("   ✅ PaddleNLP 模型加载完成")
    
    def _encode(self, texts):
        """将文本列表转为向量，返回 List[List[float]]"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pd"
        )
        
        with paddle.no_grad():
            outputs = self.model(**inputs)
            
            if isinstance(outputs, tuple):
                pooled_output = outputs[0]
            else:
                pooled_output = outputs.last_hidden_state[:, 0, :]
            
            # 转换为 list，确保每个元素是 float
            embeddings = pooled_output.numpy().astype(float).tolist()
            return embeddings
    
    def __call__(self, input):
        return self._encode(input)
    
    def embed_query(self, input):
        """Chroma 查询时调用的方法（参数名必须是 input）"""
        result = self._encode([input])
        return result[0] if result else None

# ==================== 【步骤1：切分】 ====================
def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            for sep in ["。", "！", "？", "\n\n", "\n", ". ", "! ", "? ", "; ", "，", " ", ""]:
                last_sep = text.rfind(sep, start, end)
                if last_sep != -1 and last_sep > start + chunk_size // 2:
                    end = last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_len else text_len
    
    return chunks

# ==================== 加载所有书籍 ====================
def load_and_chunk_books():
    all_chunks = []
    chunk_metadata = []
    
    for book_filename in SELECTED_BOOKS:
        book_path = PROCESSED_DIR / book_filename
        
        if not book_path.exists():
            print(f"   ⚠️ 文件不存在: {book_filename}")
            continue
        
        print(f"   📖 正在加载: {book_filename}")
        
        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "source": book_filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        print(f"      切分为 {len(chunks)} 个片段")
    
    return all_chunks, chunk_metadata

# ==================== 【步骤2：向量化 + 步骤3：存储】 ====================
def create_new_collection(client, chunks, metadatas):
    # 使用 Chroma 默认的 Embedding 函数（支持中文）
    from chromadb.utils import embedding_functions
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = client.create_collection(
        name="ai_coach",
        embedding_function=embedding_fn
    )
    
    batch_size = 50
    total = len(chunks)
    
    print(f"   🧠 正在向量化并存入 {total} 个片段...")
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_chunks = chunks[i:end]
        batch_metadatas = metadatas[i:end]
        
        ids = [f"chunk_{j}" for j in range(i, end)]
        
        metadatas_for_chroma = [
            {"source": m["source"], "chunk_index": m["chunk_index"]}
            for m in batch_metadatas
        ]
        
        # 显式计算 embeddings，并确保格式为 List[List[float]]
        batch_embeddings = embedding_fn(batch_chunks)
        
        # 检查并修正格式
        if batch_embeddings and isinstance(batch_embeddings[0], list) and len(batch_embeddings[0]) > 0 and isinstance(batch_embeddings[0][0], list):
            # 如果是三层嵌套，展平为两层
            batch_embeddings = [emb[0] for emb in batch_embeddings]
        
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=metadatas_for_chroma,
            ids=ids
        )
        
        print(f"      已存入 {end}/{total} 个片段")
    
    return collection


def build_vector_store(chunks, metadatas):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    existing_collections = client.list_collections()
    
    if "ai_coach" in existing_collections:
        print("   📂 获取已有 collection: ai_coach")
        collection = client.get_collection("ai_coach")
        count = collection.count()
        print(f"   📊 已有 {count} 个向量")
        
        if count == 0:
            print("   ⚠️ collection 为空，删除并重建...")
            client.delete_collection("ai_coach")
            return create_new_collection(client, chunks, metadatas)
        else:
            print(f"   ✅ 使用已有 collection（{count} 个向量）")
            return collection
    else:
        print("   🆕 创建新 collection: ai_coach")
        return create_new_collection(client, chunks, metadatas)

# ==================== 测试检索 ====================
def test_retrieval(collection):
    print("\n" + "=" * 50)
    print("🔍 测试检索效果")
    print("=" * 50)
    
    test_queries = [
        "排球运动员的力量训练",
        "深蹲的标准动作",
        "营养补充建议",
    ]
    
    for query in test_queries:
        print(f"\n📝 问题: {query}")
        try:
            results = collection.query(query_texts=[query], n_results=2)
            
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0], 1):
                    source = results['metadatas'][0][i-1].get('source', '未知')
                    print(f"   结果 {i}: ({source[:40]}...)")
                    print(f"   {doc[:150]}...")
                    print()
        except Exception as e:
            print(f"   ❌ 查询出错: {e}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Phase 1: 构建向量库 (PaddleNLP)")
    print("=" * 50)
    
    print(f"\n📚 选定 {len(SELECTED_BOOKS)} 本书:")
    for book in SELECTED_BOOKS:
        print(f"   - {book}")
    
    print("\n📖 【步骤1：切分】加载并切分文本...")
    chunks, metadatas = load_and_chunk_books()
    
    if not chunks:
        print("❌ 没有加载到任何文本")
        sys.exit(1)
    
    print(f"\n✅ 共切分为 {len(chunks)} 个片段")
    print(f"   📊 平均片段大小: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
    
    print("\n🧠 【步骤2：向量化】+ 【步骤3：存储】构建向量库...")
    
    if Path(CHROMA_DIR).exists():
        print("   📂 使用已有向量库（跳过重建）")
    else:
        print("   📂 首次创建向量库")
    
    collection = build_vector_store(chunks, metadatas)
    
    print(f"\n✅ 向量库构建完成！")
    print(f"   📂 保存至: {CHROMA_DIR}")
    print(f"   📊 总计: {collection.count()} 个向量")
    
    test_retrieval(collection)
    
    print("\n🎉 Phase 1 完成！")