import chromadb
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

print("=" * 50)
print("🚀 开始迁移数据：Chroma → Milvus (修正版)")
print("=" * 50)

# 1. 连接 Milvus
connections.connect(host="localhost", port="19530")
print("✅ 已连接 Milvus")

# 2. 从 Chroma 读取数据
chroma_client = chromadb.PersistentClient(path="chroma_db")
chroma_collection = chroma_client.get_collection("ai_coach")

print("📖 正在从 Chroma 读取数据...")
data = chroma_collection.get(include=["embeddings", "documents", "metadatas"])

ids = data['ids']
embeddings = data['embeddings']
documents = data['documents']
metadatas = data['metadatas']

print(f"   读取到 {len(ids)} 条记录")
print(f"   Embeddings 维度: {len(embeddings[0])}")

# 3. 在 Milvus 中创建 Collection
collection_name = "ai_coach_milvus"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"🗑️  已删除旧 collection: {collection_name}")

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields, description="AI Coach Knowledge Base")
collection = Collection(collection_name, schema)
print(f"✅ 已创建 Collection: {collection_name}")

# 4. 分批插入数据
print("📥 正在插入数据到 Milvus...")
batch_size = 100
total = len(ids)

for i in range(0, total, batch_size):
    end = min(i + batch_size, total)
    
    # 准备当前批次数据
    batch_ids = ids[i:end]
    batch_embeddings = embeddings[i:end]
    batch_documents = documents[i:end]
    batch_sources = [m.get('source', '') for m in metadatas[i:end]]
    
    # 插入
    collection.insert([
        batch_ids,
        batch_embeddings,
        batch_documents,
        batch_sources
    ])
    print(f"   已插入 {end}/{total} 条")

print("✅ 数据插入完成！")

# 5. 刷新统计
collection.flush()
print("✅ Flush 完成！")

# 6. 创建索引
print("🔧 正在创建索引...")
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)
print("✅ 索引创建完成！")

# 7. 加载到内存
print("📤 正在加载到内存...")
collection.load()
print("✅ 加载完成！")

# 8. 验证
print("=" * 50)
print("📊 迁移完成！")
print(f"   Collection: {collection_name}")
print(f"   向量总数: {collection.num_entities}")
print("=" * 50)