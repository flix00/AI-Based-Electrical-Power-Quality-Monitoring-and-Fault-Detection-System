import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from power_quality import (
    CLASSES,
    CYCLES,
    FEATURE_NAMES,
    FREQUENCY,
    NOMINAL_RMS,
    SAMPLE_RATE,
    cycle_rms,
    extract_features,
    generate_signal,
    harmonic_levels,
    train_and_evaluate,
)

st.set_page_config(page_title="Power Quality Monitor", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Training the simulated-signal classifier...")
def load_model():
    return train_and_evaluate()


model, accuracy, confusion, train_count, test_count = load_model()

st.title("⚡ Electrical Power Quality Monitor")
st.caption("50 Hz, 230 V RMS simulated voltage · Random Forest fault detection")

with st.sidebar:
    st.header("Signal settings")
    chosen = st.selectbox("Voltage condition", CLASSES)
    strength = st.slider("Disturbance severity", 0, 100, 50, 5, disabled=chosen == "Normal")
    noise = st.slider("Signal-to-noise ratio (dB)", 25, 55, 40, 1)
    seed = st.number_input("Simulation seed", min_value=0, max_value=1000000, value=42, step=1)

rng = np.random.default_rng(int(seed))
time, waveform = generate_signal(chosen, rng, severity=strength, snr_db=noise)
features = extract_features(waveform)
predicted = model.predict(features.reshape(1, -1))[0]
probabilities = model.predict_proba(features.reshape(1, -1))[0]
confidence = float(np.max(probabilities))

analysis_tab, validation_tab = st.tabs(["Signal analysis", "Model validation"])

with analysis_tab:
    left, middle, right = st.columns(3)
    left.metric("Simulated condition", chosen)
    middle.metric("Predicted condition", predicted)
    right.metric("Model confidence", f"{confidence:.1%}")

    if predicted != chosen:
        st.warning("The model classified this simulated waveform differently from its generated condition.")

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(time, waveform, linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Voltage (V)", title="Simulated voltage waveform")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    rms = cycle_rms(waveform)
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot((np.arange(CYCLES) + 0.5) / FREQUENCY, rms, marker="o", markersize=3)
    ax.axhline(NOMINAL_RMS, linestyle="--", linewidth=1, label="Nominal RMS")
    ax.set(xlabel="Time (s)", ylabel="RMS voltage (V)", title="Cycle-by-cycle RMS voltage")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        levels = harmonic_levels(waveform)
        fig, ax = plt.subplots(figsize=(6, 3))
        harmonics = np.arange(1, 10)
        ax.bar(harmonics * FREQUENCY, levels * 100)
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="Magnitude (% of fundamental)",
            title="Harmonic spectrum",
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Measured features")
        st.dataframe(
            {"Feature": FEATURE_NAMES, "Value": np.round(features, 3)},
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Class probabilities")
        st.dataframe(
            {"Class": model.classes_, "Probability (%)": np.round(probabilities * 100, 2)},
            hide_index=True,
            use_container_width=True,
        )

with validation_tab:
    first, second, third = st.columns(3)
    first.metric("Synthetic test accuracy", f"{accuracy:.1%}")
    second.metric("Training waveforms", f"{train_count:,}")
    third.metric("Held-out waveforms", f"{test_count:,}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(confusion, cmap="Blues")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set(xlabel="Predicted condition", ylabel="True condition", title="Held-out confusion matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.info("Training and testing use only simulated voltage waveforms. Accuracy on real-world electrical data is unknown.")
