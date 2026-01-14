# wearable_agent/main.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

# Load model files relative to this module's directory
_BASE_DIR = Path(__file__).resolve().parent

# Try to load model files; if missing, switch to a safe heuristic fallback
try:
    def _load_model_file(name: str):
        path = _BASE_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path}\n"
                "Put the model files (stress_model_xgb.pkl, scaler.pkl, label_encoder.pkl) "
                "inside the `wearable_agent` folder or update the path in the code."
            )
        return joblib.load(path)

    clf = _load_model_file('stress_model_xgb.pkl')
    scaler = _load_model_file('scaler.pkl')
    le = _load_model_file('label_encoder.pkl')
    _HEURISTIC_FALLBACK = False
except Exception:
    clf = None
    scaler = None
    class _LE:
        classes_ = np.array(["Low", "Mild", "Moderate", "High"])  
        def inverse_transform(self, arr):
            return np.array([self.classes_[int(i)] for i in arr])
    le = _LE()
    _HEURISTIC_FALLBACK = True

severity_order = ["Low", "Mild", "Moderate", "High"]
label_to_rank = {label: i for i, label in enumerate(severity_order)}
rank_to_label = {i: label for i, label in enumerate(severity_order)}

# Map label names to their index within the label encoder's class order
_CLASS_TO_INDEX = {label: i for i, label in enumerate(le.classes_)}

def _apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature scaling on a probability vector without logits.

    Since we don't have logits here, approximate by raising to a power and renormalizing:
    p_i' = p_i ** (1/T) / sum_j p_j ** (1/T)

    - T < 1.0 sharpens the distribution (more confident)
    - T > 1.0 flattens the distribution (more uncertain)
    """
    if temperature is None or temperature <= 0:
        # Guard against invalid temperatures; fall back to no-op
        temperature = 1.0
    if temperature == 1.0:
        return probs.copy()
    alpha = 1.0 / float(temperature)
    with np.errstate(divide='ignore', invalid='ignore'):
        powered = np.power(np.clip(probs, 1e-12, 1.0), alpha)
    total = powered.sum()
    if not np.isfinite(total) or total <= 0:
        return probs.copy()
    return powered / total

def _apply_class_boost(probs: np.ndarray, class_index: int, factor: float) -> np.ndarray:
    """Boost one class probability by a factor and renormalize.

    Args:
        probs: shape (K,) probability vector summing to 1.
        class_index: index to boost.
        factor: > 0; 1.0 means no change.
    """
    if factor is None or factor <= 0:
        factor = 1.0
    if factor == 1.0:
        return probs.copy()
    boosted = probs.copy()
    boosted[class_index] = boosted[class_index] * factor
    s = boosted.sum()
    if s <= 0 or not np.isfinite(s):
        return probs.copy()
    return boosted / s

def analyze_wearable(
    data: dict,
    proba_temperature: Optional[float] = None,
    adjusted_class_boost: Optional[float] = None,
):
    """Analyze wearable metrics and return stress category and probabilities.

    Args:
        data: Dict of input features with keys: 'avg_HR', 'avg_sleep_duration',
              'avg_sleep_efficiency', 'avg_waso', 'daily_steps', 'daily_calories'.
        proba_temperature: Optional temperature for probability scaling. If None,
              an automatic temperature is chosen based on detected extremes.
              T < 1 sharpens, T > 1 flattens, T == 1 is no-op.
        adjusted_class_boost: Optional multiplicative boost for the adjusted class.
              If None, an automatic factor is chosen based on detected extremes.

    Returns:
        Dict containing predicted_category, adjusted 'probabilities' (percent), and
        'raw_probabilities' (percent) emitted by the classifier before adjustment.
    """
    df = pd.DataFrame([data])
    
    numerical_features = ['avg_HR', 'avg_sleep_duration', 'avg_sleep_efficiency',
                          'avg_waso', 'daily_steps', 'daily_calories']

    if _HEURISTIC_FALLBACK:
        # Simple heuristic if model assets are missing
        hr = float(df['avg_HR'].iloc[0])
        sleep = float(df['avg_sleep_duration'].iloc[0])
        eff = float(df['avg_sleep_efficiency'].iloc[0]) if 'avg_sleep_efficiency' in df else 0.8
        waso = float(df['avg_waso'].iloc[0]) if 'avg_waso' in df else 50

        score = 0.0
        score += 0.4 if hr > 120 else (0.2 if hr > 100 else 0.0)
        score += 0.3 if sleep < 5 else (0.15 if sleep < 6 else 0.0)
        score += 0.2 if eff < 0.6 else (0.1 if eff < 0.7 else 0.0)
        score += 0.2 if waso > 60 else (0.1 if waso > 50 else 0.0)

        if score >= 0.7:
            pred_class = 'High'
        elif score >= 0.4:
            pred_class = 'Moderate'
        elif score >= 0.2:
            pred_class = 'Mild'
        else:
            pred_class = 'Low'

        # Create a soft probability distribution consistent with pred_class
        base = np.array([0.7, 0.2, 0.07, 0.03]) 
        idx = {"Low": 0, "Mild": 1, "Moderate": 2, "High": 3}[pred_class]
        base = np.roll(base, idx)
        pred_proba = base / base.sum()
    else:
        # Scale numerical features
        df_scaled = scaler.transform(df[numerical_features])
        # Model prediction
        pred_proba = clf.predict_proba(df_scaled)[0]
        pred_num = np.argmax(pred_proba)
        pred_class = le.inverse_transform([pred_num])[0]
    
    # Extreme condition adjustment
    current_rank = label_to_rank[pred_class]
    extreme_flags = [
        df['avg_HR'].iloc[0] > 140,
        df['avg_sleep_duration'].iloc[0] < 5,
        df['avg_sleep_efficiency'].iloc[0] < 0.6,
        df['avg_waso'].iloc[0] > 60
    ]
    extreme_count = sum(extreme_flags)
    
    if df['avg_HR'].iloc[0] > 150 or df['avg_sleep_duration'].iloc[0] < 4:
        adjusted_rank = label_to_rank["High"]
        adjust_strength = "strong"
    elif extreme_count >= 2:
        adjusted_rank = label_to_rank["High"]
        adjust_strength = "strong"
    elif extreme_count == 1:
        adjusted_rank = min(current_rank + 1, max(label_to_rank.values()))
        adjust_strength = "mild"
    else:
        adjusted_rank = current_rank
        adjust_strength = "none"

    pred_class_adjusted = rank_to_label[adjusted_rank]

    # --- Adjust probabilities in tandem with severity/category adjustment ---
    # Strategy: First boost the adjusted class significantly, then apply temperature
    # Choose defaults, allow explicit overrides
    if adjusted_class_boost is not None:
        boost_factor = adjusted_class_boost
    else:
        if adjust_strength == "strong":
            boost_factor = 50.0  # very significantly emphasize adjusted class
        elif adjust_strength == "mild":
            boost_factor = 3.0  # moderately emphasize
        else:
            boost_factor = 1.0

    if proba_temperature is not None:
        temperature = proba_temperature
    else:
        if adjust_strength == "strong":
            temperature = 0.8  # sharpen after boosting
        elif adjust_strength == "mild":
            temperature = 0.9  # slightly sharpen
        else:
            temperature = 1.0

    # First boost the adjusted class
    idx_adjusted = _CLASS_TO_INDEX.get(pred_class_adjusted, None)
    if idx_adjusted is not None:
        proba_boosted = _apply_class_boost(pred_proba, idx_adjusted, boost_factor)
    else:
        proba_boosted = pred_proba
    
    # apply temperature scaling to sharpen the boosted distribution
    proba_adjusted = _apply_temperature(proba_boosted, temperature)

    # Prepare probability dictionaries 
    raw_proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(le.classes_, pred_proba)}
    proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(le.classes_, proba_adjusted)}

    return {
        "agent": "wearable_agent",
        "predicted_category": pred_class_adjusted,
        "probabilities": proba_dict,
    }

# Example:
if __name__ == "__main__":
    sample_data = {
        'avg_HR': 145,
        'avg_sleep_duration': 4.5,
        'avg_sleep_efficiency': 0.55,
        'avg_waso': 70,
        'daily_steps': 2000,
        'daily_calories': 1400
    }
    # Use automatic probability adjustment based on detected extremes
    # (or pass proba_temperature and adjusted_class_boost to override)
    result = analyze_wearable(sample_data)
    
    # Display as formatted JSON
    import json
    print(json.dumps(result, indent=2))
