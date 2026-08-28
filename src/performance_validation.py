import time
import sys
from datetime import datetime
from generation_milvus import retrieve as retrieve_milvus
from generation_hyde_rerank import retrieve_hyde_rerank

questions = [
    "排球运动员和一般运动员在体能上有哪些独特的优势？",
    "排球运动员和田径运动员在运动能力上有哪些方面是类似的？哪些方面是不同的？",
    "哪些弹跳训练对排球运动员是特别合适的？",
    "对于弹跳力的训练和运动，在热身、放松以及伤病管理上应该注意什么？",
    "排球运动员可以向健美运动员学习哪些方面内容？",
    "排球运动员在饮食上有哪些需要注意的地方？",
    "哪些田径训练可以辅助排球的训练？",
    "排球运动员想全方位地提高爆发力，可以做哪些训练？",
    "排球运动员应该如何对待野球(pick-up game)？",
    "非赛季期间，排球运动员应该怎样控制运动的强度？",
    "除了扣球、拦网、后排进攻、大力跳发球的动作模式，排球运动员还可以做哪些类似的跳跃运动来提升自己的弹跳力和爆发力？",
]

# ==================== 日志 ====================
log_filename = f"docs/phase35_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

print("=" * 70)
print("Phase 3.5: Performance Validation (优化版)")
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

def clean_text(text, max_len=300):
    """清理换行和多余空格，截断到指定长度"""
    cleaned = ' '.join(text.split())
    return cleaned[:max_len] + "..." if len(cleaned) > max_len else cleaned

for i, q in enumerate(questions, 1):
    print(f"\n[{i}] {q}")
    print("-" * 50)
    
    # ===== Phase 2 =====
    docs, sources = retrieve_milvus(q, n_results=1)
    print("\n  Phase 2 (Milvus):")
    if docs:
        print(f"    来源: {sources[0]}")
        print(f"    片段: {clean_text(docs[0], 300)}")
    else:
        print("    未找到")
    
    # ===== Phase 3 =====
    docs, sources, scores, _ = retrieve_hyde_rerank(q, top_k=1)
    print("\n  Phase 3 (HyDE + Rerank):")
    if docs:
        print(f"    来源: {sources[0]}")
        print(f"    分数: {scores[0]:.4f}")
        print(f"    片段: {clean_text(docs[0], 300)}")
    else:
        print("    未找到")
    
    print("-" * 50)
    time.sleep(0.5)

print("\n" + "=" * 70)
print(f"✅ 完成。结果已保存至: {log_filename}")
print("=" * 70)

log_file.close()
sys.stdout = sys.__stdout__
print(f"结果已保存至: {log_filename}")