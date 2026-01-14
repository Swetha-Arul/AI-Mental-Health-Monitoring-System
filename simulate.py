import gradio as gr
from fusion_agent.main import fuse_agents

def simulate(text, hr, sleep, steps, calories):
    if not text or text.strip() == "":
        return "⚠️ Please enter your text input — describe how you're feeling today."
    record = fuse_agents(
        text_input=text,
        avg_HR=hr,
        avg_sleep_duration=sleep,
        daily_steps=steps,
        daily_calories=calories
    )

    result = record["fusion_result"]
    output_text = (
        f"🕒 Timestamp: {record['timestamp']}\n\n"
        f"💬 Text Analysis: {record['text_agent_output']['label']}\n"
        f"⌚ Wearable Analysis: {record['wearable_agent_output']['predicted_category']}\n\n"
        f"⚠️ Risk Level: {result['status']}\n"
        f"📊 Risk Score: {result['risk_score']}\n\n"
        f"🧠 {result['interpretation']}\n\n"
        f"🔍  {result['explanation']}"
    )
    return output_text

ui = gr.Interface(
    fn=simulate,
    inputs=[
        gr.Textbox(label="Social Media / Journal Text", placeholder="Type something like 'I feel tired and unmotivated today'"),
        gr.Slider(50, 160, step=1, label="Average Heart Rate (bpm)"),
        gr.Slider(3, 10, step=0.1, label="Average Sleep Duration (hours)"),
        gr.Slider(1000, 15000, step=500, label="Daily Steps"),
        gr.Slider(800, 3000, step=50, label="Daily Calories Burned"),
    ],
    outputs=gr.Textbox(label="Simulation Result", lines=10),
    title="🧠 Fusion Health Simulation",
    description="Simulates emotional and physiological data fusion to assess mental health risk.",
)

ui.launch(share=True)
