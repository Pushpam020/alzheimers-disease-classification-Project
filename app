# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

st.set_page_config(page_title="Solar Power Generation Predictor", layout="wide")

# -----------------------
# Helpers
# -----------------------
def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}. Make sure it's uploaded to the repo.")
        st.stop()
    return joblib.load(path)

def predict_power(model, scaler, X_raw):
    """Scale input and return numeric prediction (float)."""
    X_scaled = scaler.transform(np.array(X_raw).reshape(1, -1))
    y_pred = model.predict(X_scaled)
    # If model returns array of shape (1,), get value
    if hasattr(y_pred, "__len__"):
        return float(y_pred[0])
    return float(y_pred)

def power_level_color(value, thresholds=(1000, 3000)):
    """Return (label, color_hex) for power value.
       thresholds: (low_to_moderate, moderate_to_high) """
    low_th, high_th = thresholds
    if value < low_th:
        return "Low", "#ff6b6b"        # red-ish
    elif value < high_th:
        return "Moderate", "#ffc857"   # yellow-ish
    else:
        return "High", "#4caf50"       # green-ish

def draw_power_bar(pred_value, max_value=5000):
    """Draw horizontal bar with value label using matplotlib and return fig."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.barh([0], [pred_value], height=0.6, color=current_color, edgecolor="k")
    ax.set_xlim(0, max_value)
    ax.set_yticks([0])
    ax.set_yticklabels(["Predicted Power"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # place value text in middle of bar (or at end if bar too small)
    xpos = pred_value / 2 if pred_value > 150 else pred_value + 50
    ax.text(xpos, 0, f"{int(round(pred_value))}", va="center", ha="center", fontsize=12, color="black", fontweight="bold")
    ax.set_xlabel("Power (Units)")
    ax.set_frame_on(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout()
    return fig

def draw_trend_chart(base_inputs, model, scaler, steps=5, delta=0.05):
    """Create small trend by perturbing a single important input (distance-to-solar-noon)."""
    # We'll vary distance-to-solar-noon slightly around current value
    idx_distance = 0  # index of distance-to-solar-noon in feature order
    base = np.array(base_inputs, dtype=float)
    distances = np.linspace(max(0, base[idx_distance] - delta), min(1, base[idx_distance] + delta), steps)
    preds = []
    for d in distances:
        tmp = base.copy()
        tmp[idx_distance] = d
        preds.append(predict_power(model, scaler, tmp))
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(distances, preds, marker="o")
    ax.set_xlabel("distance-to-solar-noon")
    ax.set_ylabel("Power (units)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return fig

# -----------------------
# Load model + scaler
# -----------------------
with st.spinner("Loading model and scaler..."):
    try:
        model = load_pickle("best_model.pkl")   # your trained Gradient Boosting model
        scaler = load_pickle("scaler.pkl")      # your StandardScaler used at training
    except Exception as e:
        st.error("Error loading model or scaler. See logs.")
        raise

# -----------------------
# Sidebar: Inputs
# -----------------------
st.sidebar.header("Input Parameters")

# defaults
defaults = {
    "distance-to-solar-noon": 0.50,
    "temperature": 70,
    "wind-direction": 90,
    "wind-speed": 5.0,
    "sky-cover": 20,
    "visibility": 10.0,
    "humidity": 50,
    "average-wind-speed-(period)": 5.0,
    "average-pressure-(period)": 29.8
}

# init session state for reset functionality
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_inputs():
    for k, v in defaults.items():
        st.session_state[k] = v

st.sidebar.number_input("distance-to-solar-noon (0–1)", min_value=0.00, max_value=1.00, value=st.session_state["distance-to-solar-noon"], step=0.01, key="distance-to-solar-noon", format="%.2f")
st.sidebar.number_input("temperature (°F)", min_value=-50, max_value=150, value=int(st.session_state["temperature"]), step=1, key="temperature")
st.sidebar.number_input("wind-direction (deg)", min_value=0, max_value=360, value=int(st.session_state["wind-direction"]), step=1, key="wind-direction")
st.sidebar.number_input("wind-speed (mph)", min_value=0.0, max_value=100.0, value=float(st.session_state["wind-speed"]), step=0.1, key="wind-speed", format="%.2f")
st.sidebar.number_input("sky-cover (0–100)", min_value=0, max_value=100, value=int(st.session_state["sky-cover"]), step=1, key="sky-cover")
st.sidebar.number_input("visibility (miles)", min_value=0.0, max_value=100.0, value=float(st.session_state["visibility"]), step=0.1, key="visibility", format="%.2f")
st.sidebar.number_input("humidity (%)", min_value=0, max_value=100, value=int(st.session_state["humidity"]), step=1, key="humidity")
st.sidebar.number_input("average-wind-speed-(period)", min_value=0.0, max_value=50.0, value=float(st.session_state["average-wind-speed-(period)"]), step=0.1, key="average-wind-speed-(period)", format="%.2f")
st.sidebar.number_input("average-pressure-(period)", min_value=0.0, max_value=200.0, value=float(st.session_state["average-pressure-(period)"]), step=0.01, key="average-pressure-(period)", format="%.2f")

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Reset to Defaults"):
    reset_inputs()

# -----------------------
# Main UI
# -----------------------
st.title("☀️ Solar Power Generation Predictor")
st.write("Enter the weather parameters (left) and click **Predict** to get an estimated power output.")

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("🔮 Predict")
with col2:
    st.write("")  # spacing

# Build feature vector in the trained order
# IMPORTANT: feature order MUST match training order
feature_order = [
    "distance-to-solar-noon",
    "temperature",
    "wind-direction",
    "wind-speed",
    "sky-cover",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)"
]

X_input = [st.session_state[name] for name in feature_order]

# Predict on click or show message
pred_value = None
if predict_clicked:
    try:
        pred_value = predict_power(model, scaler, X_input)
    except Exception as e:
        st.error("Prediction error — check model, scaler and feature order.")
        st.exception(e)

# If not clicked, you can still preview a default prediction (optional)
if pred_value is None:
    pred_value = predict_power(model, scaler, X_input)

# Power level and color
level_label, current_color = power_level_color(pred_value, thresholds=(1200, 3000))

# Show numeric result
st.success(f"Estimated Power Generated: {int(round(pred_value)):,} units")

# Visualization: big bar
st.markdown("### 🌤️ Power Generation Visualization")
fig_bar = draw_power_bar(pred_value, max_value=5000)
st.pyplot(fig_bar)

# Level box
if level_label == "Low":
    st.warning(f"⚡ Power Level: **{level_label}** — Prediction indicates low power generation under current weather conditions.")
elif level_label == "Moderate":
    st.info(f"⚡ Power Level: **{level_label}** — Prediction indicates moderate power generation.")
else:
    st.success(f"⚡ Power Level: **{level_label}** — Prediction indicates high power generation.")

# Mini trend
st.markdown("### 📈 Power Comparison Trend")
fig_trend = draw_trend_chart(X_input, model, scaler, steps=5, delta=0.05)
st.pyplot(fig_trend)

# Show input table
with st.expander("🔧 See input as table"):
    df_inputs = pd.DataFrame([X_input], columns=feature_order)
    st.dataframe(df_inputs.T.rename(columns={0: "value"}))

st.markdown("---")

# Notes & About
st.markdown("#### Notes")
st.markdown("- This app uses the trained **Gradient Boosting model and StandardScaler**.")
st.markdown("- Feature order must match the training sequence exactly.")
st.markdown("- Make sure `best_model.pkl` and `scaler.pkl` are uploaded to the application folder.")

with st.expander("ℹ️ About this app"):
    st.write(
        "This Streamlit app loads a pre-trained Gradient Boosting regressor and StandardScaler.\n\n"
        "It accepts weather inputs from the sidebar, scales them the same way as during training, "
        "and returns a power prediction. The visualization includes a color-coded bar and a small trend preview."
    )

with st.expander("❓ What do the inputs mean?"):
    st.write(
        "- **distance-to-solar-noon (0–1):** fraction where 0=midnight and 0.5=solar noon (higher -> closer to noon)\n"
        "- **temperature (°F):** air temperature\n"
        "- **wind-direction (deg):** wind direction in degrees\n"
        "- **wind-speed (mph):** wind speed\n"
        "- **sky-cover (0–100):** cloud coverage percentage\n"
        "- **visibility (miles):** visibility distance\n"
        "- **humidity (%):** humidity percentage\n"
        "- **average-wind-speed-(period):** average wind speed during measurement period\n"
        "- **average-pressure-(period):** average atmospheric pressure during period"
    )

# Footer
st.markdown("---")
st.markdown("🧾 Built by Pushpam Kumari | Model: Gradient Boosting | Deployed on Streamlit Cloud 🌐")
