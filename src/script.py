import os
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel

os.environ["CHROMA_ONNX_DISABLE"] = "true"

# 加载模型
model_name = "rocketqa-zh-dureader-query-encoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def encode(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pd"
    )
    
    with paddle.no_grad():
        outputs = model(**inputs)
        if isinstance(outputs, tuple):
            pooled_output = outputs[0]
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output.numpy().astype(float).tolist()

# 测试
test_text = "排球运动员的力量训练"
result = encode(test_text)
print(f"文本: {test_text}")
print(f"向量维度: {len(result[0]) if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else '未知'}")
print(f"向量类型: {type(result)}")
print(f"结果结构: {len(result)} 个元素，每个元素是 {type(result[0])}")