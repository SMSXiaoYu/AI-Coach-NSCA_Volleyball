"""
Phase 5: 代码壳 + Dify脑 混合架构示例
======================================

核心思想：
  原有 FastAPI 后端 (api/main.py) 保持不变 → 代码壳
  核心 RAG 逻辑 (检索+生成) 改为调用 Dify API → Dify脑

Dify 提供两种 API：
  1. Dify API (应用API) — 给外部应用调用，用 API Key 认证，最适合集成
  2. Console API (控制台API) — 管理用途，需要前端加密或 cookie

本示例演示用 Dify API 实现 RAG 检索 + 问答，替换原有自研 RAG 链路。

使用前准备：
  1. 确保 Dify 运行在 http://localhost
  2. 在 Dify 界面创建知识库 + 文档
  3. 创建一个 Chatflow 应用，关联该知识库
  4. 从应用详情页获取 API Key

运行：
  python src/phase5_dify_integration.py
"""

import json
import urllib.request
import urllib.error

# ================== 配置区 ==================
# Dify 入口地址（通过 nginx 统一访问）
DIFY_BASE = "http://localhost"

# ===== 方式 A：Dify API（应用 API，推荐）=====
# 需要在 Dify 界面创建 Chatflow 应用并获取 API Key
# 设置位置：应用详情 → API 访问 → 生成 API Key
DIFY_APP_API_KEY = "app-xxxxxxxxxxxxxxxxxxxxxxxx"  # TODO: 替换为你的 API Key

# ===== 方式 B：Console API（控制台 API，管理用途）=====
# 需要 setup/admin 创建账户后，用 JWT token 认证
# 可用 api 容器内 Flask shell 生成 token
DIFY_CONSOLE_TOKEN = ""  # TODO: 填入 Console API token


# ================== Dify API 封装 ==================

class DifyClient:
    """Dify API 客户端 —— Phase 5 核心集成层"""

    def __init__(self, base_url=DIFY_BASE, app_api_key=None, console_token=None):
        self.base = base_url
        self.app_key = app_api_key
        self.console_token = console_token

    # ---------- 1. Dify API（应用层，替代自研 generation.py）----------

    def chat(self, query: str, conversation_id: str = "") -> dict:
        """
        调用 Dify Chatflow 应用进行问答
        替代：原 generation.py 的完整 RAG 流程

        POST /v1/chat-messages
        """
        body = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",  # blocking 或 streaming
            "conversation_id": conversation_id,
            "user": "ai-coach-user"
        }
        return self._request(
            "POST", f"{self.base}/v1/chat-messages",
            body=body, api_key=self.app_key
        )

    def completion(self, prompt: str) -> dict:
        """
        调用 Dify Text Generator 应用
        """
        body = {
            "inputs": {},
            "query": prompt,
            "response_mode": "blocking",
            "user": "ai-coach-user"
        }
        return self._request(
            "POST", f"{self.base}/v1/completion-messages",
            body=body, api_key=self.app_key
        )

    def list_conversations(self) -> dict:
        """列出当前用户的会话"""
        return self._request(
            "GET", f"{self.base}/v1/conversations?user=ai-coach-user",
            api_key=self.app_key
        )

    # ---------- 2. Console API（管理层，替代自研数据管道）----------

    def list_datasets(self) -> dict:
        """列出所有知识库"""
        return self._request(
            "GET", f"{self.base}/console/api/datasets",
            console_token=self.console_token
        )

    def create_dataset(self, name: str, description: str = "") -> dict:
        """创建知识库"""
        return self._request(
            "POST", f"{self.base}/console/api/datasets",
            body={"name": name, "description": description, "provider": "vendor"},
            console_token=self.console_token
        )

    def create_document(self, dataset_id: str, name: str, text: str) -> dict:
        """向知识库添加文档（文本分段自动处理）"""
        return self._request(
            "POST", f"{self.base}/console/api/datasets/{dataset_id}/documents",
            body={
                "indexing_mode": "high_quality",
                "dataset_processing_rule": {"mode": "automatic"},
                "document": {"name": name, "text": text}
            },
            console_token=self.console_token
        )

    def retrieve(self, dataset_id: str, query: str, top_k: int = 3) -> dict:
        """
        检索知识库
        替代：原 retrieval_hyde_rerank.py
        """
        return self._request(
            "POST", f"{self.base}/console/api/datasets/{dataset_id}/retrieve",
            body={
                "query": query,
                "retrieval_mode": "semantic_search",
                "top_k": top_k
            },
            console_token=self.console_token
        )

    # ---------- 底层 HTTP ----------

    def _request(self, method, url, body=None, api_key=None, console_token=None):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if console_token:
            headers["Authorization"] = f"Bearer {console_token}"

        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body_text)
            except Exception:
                return {"error": body_text, "status": e.code}


# ================== 使用示例 ==================

def demo_architecture():
    """展示 Phase 5 混合架构的数据流"""
    print("=" * 60)
    print("Phase 5: 代码壳 + Dify脑 架构演示")
    print("=" * 60)
    print()
    print("""
┌─────────────────────┐        ┌──────────────────────────┐
│   原有 FastAPI 后端  │        │      Dify (1.17.0)       │
│   (api/main.py)     │        │  ┌────────────────────┐  │
│                     │        │  │ Chatflow 应用       │  │
│  /ask  ←── 客户端   │──调用──▶│  │   - Hybrid RAG     │  │
│                     │        │  │   - LLM 回答       │  │
│  /retrieve          │        │  └────────────────────┘  │
│  /health            │        │  ┌────────────────────┐  │
│                     │        │  │ 知识库 (Datasets)   │  │
│  代码壳：保持不变    │        │  │   - 4本书          │  │
│  Dify脑：接管RAG     │        │  │   - 自动分段/索引  │  │
└─────────────────────┘        │  └────────────────────┘  │
                               └──────────────────────────┘
""")

    client = DifyClient(base_url=DIFY_BASE)

    # 1. 验证 Dify 健康
    print("--- Dify 版本验证 ---")
    try:
        r = urllib.request.urlopen(f"{DIFY_BASE}/console/api/version?current_version=1.17.0").read()
        print(f"✅ Dify 可达: {r.decode().strip()}")
    except Exception as e:
        print(f"❌ Dify 不可达: {e}")
        return

    # 2. 如果配置了 Console token，测试知识库管理
    if DIFY_CONSOLE_TOKEN:
        client.console_token = DIFY_CONSOLE_TOKEN
        print("\n--- Console API 链路测试 ---")

        # 列出知识库
        r = client.list_datasets()
        print(f"知识库列表: {json.dumps(r, indent=2, ensure_ascii=False)[:300]}")

        # 创建测试知识库
        r = client.create_dataset("Phase5 Test KB", "Integration test")
        ds_id = (r.get("data") or {}).get("id")
        print(f"创建知识库: {ds_id or '失败'}")

        if ds_id:
            # 添加文档
            r = client.create_document(
                ds_id, "Volleyball Guide",
                "扣球技术要点：助跑3-4米，起跳屈膝收腹，击球肘高于肩甩手腕鞭打，落地屈膝缓冲。"
            )
            print(f"添加文档: batch={r.get('data', {}).get('batch', '?')}")

            # 检索
            r = client.retrieve(ds_id, "扣球如何击球")
            print(f"检索结果: {json.dumps(r, indent=2, ensure_ascii=False)[:400]}")
    else:
        print("\n⚠️ 未配置 Console API Token，跳过知识库管理测试")
        print("   获取方法: docker exec -v ... docker-api-1 python /tmp/gen_token.py")

    # 3. 如果配置了 App API Key，测试问答
    if DIFY_APP_API_KEY and DIFY_APP_API_KEY != "app-xxxxxxxxxxxxxxxxxxxxxxxx":
        client.app_key = DIFY_APP_API_KEY
        print("\n--- Dify API (问答) 测试 ---")
        r = client.chat("排球扣球的关键动作是什么？")
        print(f"问答结果: {json.dumps(r, indent=2, ensure_ascii=False)[:400]}")
    else:
        print("\n⚠️ 未配置 Dify App API Key，跳过问答测试")
        print("   获取方法: Dify 界面 → 创建 Chatflow 应用 → API 访问 → 生成 Key")

    print("\n" + "=" * 60)
    print("Phase 5 架构理解总结：")
    print("=" * 60)
    print("""
为什么要用 Dify？
───────────────────────────────
✅ 低代码迭代：不用自己维护 RAG 管道（分段、嵌入、索引、重排）
✅ 可视化编排：工作流编辑器拖拽式连接检索/生成节点
✅ 多模型切换：一个界面管理 LLM、Embedding、Rerank 供应商
✅ 生产就绪：监控日志、API 限流、SSO、审计开箱即用
✅ 保留控制权：代码壳仍是你的，Dify 脑可随时换

集成点在哪里？
───────────────────────────────
api/main.py
  └── /ask 端点
      原来: generation_hyde_rerank.py → 本地 Milvus + LLM
      现在: requests.post('http://localhost/v1/chat-messages')

迁移成本？
───────────────────────────────
⏱ 代码量: ~50 行 client 封装（本文件）
⏱ 数据迁移: 用 Console API 把现有 4 本书导入 Dify
⏱ 学习曲线: 读一遍本文件 + 看 Dify 界面 30 分钟
""")


if __name__ == "__main__":
    demo_architecture()
