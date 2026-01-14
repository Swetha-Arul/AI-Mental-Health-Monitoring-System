import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_agent.main import analyze_text
from wearable_agent.main import analyze_wearable

severity_map = {
    "Low": 0.2, "Mild": 0.4, "Moderate": 0.7, "High": 1.0,
    "normal": 0.1, "anxiety": 0.7, "depression": 0.8, "suicidal": 1.0
}

STORAGE_FILE = Path(__file__).resolve().parent / "results_log.jsonl"


def fuse_agents(
    text_input: str,
    avg_HR: float,
    avg_sleep_duration: float,
    avg_sleep_efficiency: float = 0.8,
    avg_waso: float = 50,
    daily_steps: int = 4000,
    daily_calories: int = 1400
):
    """Fuse text and wearable agent outputs into a combined mental health risk score."""
    text_result = analyze_text(text_input)

    wearable_input = {
        "avg_HR": avg_HR,
        "avg_sleep_duration": avg_sleep_duration,
        "avg_sleep_efficiency": avg_sleep_efficiency,
        "avg_waso": avg_waso,
        "daily_steps": daily_steps,
        "daily_calories": daily_calories,
    }
    wearable_result = analyze_wearable(wearable_input)

    # normalize padded label from text agent (labels are padded to width 10)
    text_label_norm = text_result["label"].strip().lower()
    text_sev = severity_map.get(text_label_norm, 0.5)
    wearable_sev = severity_map.get(wearable_result["predicted_category"], 0.5)
    fused_risk = 0.6 * text_sev + 0.4 * wearable_sev

    if fused_risk >= 0.8:
        final_status = "High Risk"
    elif fused_risk >= 0.6:
        final_status = "Moderate Risk"
    elif fused_risk >= 0.4:
        final_status = "Mild Risk"
    else:
        final_status = "Low Risk"

    warnings = []
    advice = []

    label = text_result["label"].lower()
    hr = wearable_input["avg_HR"]
    sleep = wearable_input["avg_sleep_duration"]
    steps = wearable_input["daily_steps"]
    wearable_cat = wearable_result["predicted_category"].lower()

    # Text-based warnings
    if label in ["anxiety", "depression", "suicidal"]:
        warnings.append(f"Text shows emotional distress ({label}).")
        if label == "anxiety":
            advice.append("Resources for help with anxiety: try guided breathing apps or reach out to a counselor.")
        elif label == "depression":
            advice.append("Resources for depression: talk to trusted friends or mental-health hotlines.")
        elif label == "suicidal":
            advice.append("⚠ If you feel unsafe, please reach out immediately to a suicide helpline or local emergency number.")
    elif label in ["sad", "fear", "anger"]:
        warnings.append(f"Text reflects emotional instability ({label}).")
        advice.append("Consider journaling or mindfulness exercises to process your emotions.")

    # Wearable-based warnings
    if wearable_cat == "low":
        if hr > 120:
            warnings.append(f"Heart rate is mildly elevated ({hr} bpm) — overall wearable risk is Low, no immediate concern.")
            advice.append("Try slow breathing or short breaks if you feel stressed.")
        if sleep < 6:
            warnings.append(f"Sleep duration is slightly low ({sleep} hrs) — overall wearable risk is Low.")
            advice.append("Try to get at least 7 hours of sleep per night.")
        if steps < 5000:
            warnings.append(f"Daily steps are below 5000 — overall wearable risk is Low.")
            advice.append("Go for a short walk or stretch to improve energy.")
    else:
        if hr > 120:
            warnings.append(f"Elevated heart rate detected ({hr} bpm).")
            advice.append("Consider resting or monitoring vitals closely.")
        if sleep < 6:
            warnings.append("Short sleep duration may affect emotional stability.")
            advice.append("Try to get at least 7 hours of sleep per night.")
        if steps < 5000:
            warnings.append("Low activity levels can contribute to fatigue or low mood.")
            advice.append("Go for a 30-minute walk or stretch to improve energy.")

    comforting_messages = []

    # Text-based emotional comfort
    if label == "depression":
        comforting_messages.append("You deserve rest, care, and gentleness today — be kind to yourself, you’re doing your best. 💛")
    elif label == "anxiety":
        comforting_messages.append("It\'s okay to slow down — breathe deeply, and remind yourself you’re safe right now. 🌤")
    elif label == "suicidal":
        comforting_messages.append("You matter deeply. Please reach out to someone you trust or a local helpline — you don’t have to face this alone. ❤")

    # Wearable signals comfort
    if steps < 5000:
        comforting_messages.append("A little movement can work wonders — try a short walk or stretch to refresh your mind and body. 🌿")
    if hr > 120:
        comforting_messages.append("Your body feels tense — take a slow breath, pause for a moment, and let yourself unwind. 💆‍♀")
    if sleep < 6:
        comforting_messages.append("Rest is healing — give your body the gift of a full night’s sleep whenever you can. 🌙")

    # If overall low risk, add positive reinforcement
    if final_status == "Low Risk":
        comforting_messages.append("You’re doing great! Keep caring for yourself and celebrating small victories. 🌟")

    #Explanation 
    if warnings or advice:
        warning_text = " ".join(warnings)
        advice_text = " ".join(advice)
        explanation = f" 🟥 Warning: {warning_text}\n💡 Advice: {advice_text}\n{ ' '.join(comforting_messages) }"
    else:
        explanation = " ".join(comforting_messages)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_agent_output": text_result,
        "wearable_agent_output": wearable_result,
        "fusion_result": {
            "risk_score": round(fused_risk, 3),
            "status": final_status,
            "interpretation": f"User is at {final_status} based on emotional and physiological indicators.",
            "explanation": explanation
        }
    }
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORAGE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record

if __name__ == "__main__":
    # Example 
    sample_result = fuse_agents(
        text_input="I feel anxious and stressed today",
        avg_HR=95,
        avg_sleep_duration=7.5,
        avg_sleep_efficiency=0.85,
        avg_waso=45,
        daily_steps=8000,
        daily_calories=1800
    )
    
    print("\n=== Fusion Agent Result ===")
    print(json.dumps(sample_result, indent=2))
    print(f"\nResult saved to: {STORAGE_FILE}")
