from openai import OpenAI

client = OpenAI(
    api_key="你的DeepSeek Key",
    base_url="https://api.deepseek.com/v1"
)

try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="测试文本"
    )
    print("✅ Embedding 可用！")
    print(f"向量维度: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ Embedding 不可用: {e}")
    print("DeepSeek 不支持 Embedding，需要用替代方案")