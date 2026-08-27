import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pymilvus import connections, Collection
from chromadb.utils import embedding_functions

# 连接 Milvus
connections.connect(host="localhost", port="19530")
collection = Collection("ai_coach_milvus")
collection.load()

# 初始化 Embedding 函数
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 测试查询
test_queries = [
    "排球运动员的力量训练",
    "如何提高弹跳力",
    "营养补充建议"
]

for query in test_queries:
    print(f"\n📝 问题: {query}")
    
    # 将问题转为向量
    query_embedding = embedding_fn([query])
    
    # Milvus 搜索
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=2,
        output_fields=["text", "source"]
    )
    
    for hits in results:
        for i, hit in enumerate(hits):
            print(f"   结果 {i+1}: {hit.entity.get('source', '未知')[:40]}...")
            print(f"   {hit.entity.get('text', '')[:150]}...")
            print()