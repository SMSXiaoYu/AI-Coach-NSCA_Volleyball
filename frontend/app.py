import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/ask"

def ask_coach(question, top_k):
    if not question:
        return "请输入问题", "", ""
    
    try:
        response = requests.post(
            API_URL,
            json={"question": question, "top_k": int(top_k)},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            scores = data.get("scores", [])
            
            # 格式化来源
            source_text = "\n".join([f"📖 {s}" for s in sources]) if sources else "无来源"
            
            # 格式化分数
            score_text = "\n".join([f"📊 相关度: {s:.4f}" for s in scores]) if scores else ""
            
            return answer, source_text, score_text
        else:
            return f"❌ 服务器错误: {response.status_code}", "", ""
            
    except requests.exceptions.Timeout:
        return "⏰ 请求超时，请稍后重试", "", ""
    except Exception as e:
        return f"❌ 错误: {str(e)}", "", ""


with gr.Blocks(title="AI 排球教练", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏐 AI 排球教练
    ### 基于专业书籍的智能问答系统
    > 输入你的训练问题，AI 将从多本专业书籍中检索相关知识并生成回答
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="你的问题",
                placeholder="例如：如何提高弹跳力？",
                lines=3
            )
            
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="参考书籍数量",
                info="AI 会参考几本书籍的内容来回答"
            )
            
            submit_btn = gr.Button("🚀 获取回答", variant="primary")
        
        with gr.Column(scale=3):
            answer_output = gr.Markdown(label="AI 回答")
    
    with gr.Row():
        with gr.Column():
            source_output = gr.Textbox(label="📖 参考来源", lines=2, interactive=False)
        with gr.Column():
            score_output = gr.Textbox(label="📊 相关度分数", lines=2, interactive=False)
    
    # 绑定事件
    submit_btn.click(
        fn=ask_coach,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, source_output, score_output]
    )
    
    # 示例问题
    gr.Examples(
        examples=[
            "排球运动员如何提高弹跳力？",
            "排球运动员在饮食上需要注意什么？",
            "排球运动员如何预防膝盖伤病？",
            "排球运动员需要做哪些力量训练？"
        ],
        inputs=question_input
    )

if __name__ == "__main__":
    demo.launch(share=True)