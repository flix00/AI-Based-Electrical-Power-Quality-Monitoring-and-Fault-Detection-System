import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 6400
FREQUENCY = 50
NOMINAL_RMS = 230
CYCLES = 20
SAMPLES_PER_CYCLE = SAMPLE_RATE // FREQUENCY
N_SAMPLES = CYCLES * SAMPLES_PER_CYCLE
CLASSES = ("Normal", "Voltage sag", "Voltage swell", "Harmonic distortion")
FEATURE_NAMES = (
    "Minimum cycle RMS (p.u.)",
    "Maximum cycle RMS (p.u.)",
    "Mean cycle RMS (p.u.)",
    "Cycle RMS standard deviation (p.u.)",
    "Cycle RMS range (p.u.)",
    "3rd harmonic (%)",
    "5th harmonic (%)",
    "Total harmonic distortion (%)",
)


def generate_signal(kind, rng, severity=None, snr_db=None):
    if kind not in CLASSES:
        raise ValueError("Unknown disturbance")
    if severity is not None and not 0 <= severity <= 100:
        raise ValueError("Severity must be between 0 and 100")

    time = np.arange(N_SAMPLES) / SAMPLE_RATE
    phase = rng.uniform(0, 2 * np.pi)
    base_rms = NOMINAL_RMS * rng.uniform(0.985, 1.015)
    fundamental = np.sqrt(2) * base_rms * np.sin(2 * np.pi * FREQUENCY * time + phase)
    signal = fundamental.copy()

    if kind in ("Voltage sag", "Voltage swell"):
        start_cycle = int(rng.integers(3, 9))
        length_cycles = int(rng.integers(5, 11))
        start = start_cycle * SAMPLES_PER_CYCLE
        stop = (start_cycle + length_cycles) * SAMPLES_PER_CYCLE
        if kind == "Voltage sag":
            factor = rng.uniform(0.35, 0.75) if severity is None else 0.75 - 0.40 * severity / 100
        else:
            factor = rng.uniform(1.20, 1.55) if severity is None else 1.20 + 0.35 * severity / 100
        signal[start:stop] = fundamental[start:stop] * factor

    if kind == "Harmonic distortion":
        third = rng.uniform(0.12, 0.32) if severity is None else 0.12 + 0.20 * severity / 100
        fifth = rng.uniform(0.05, 0.17) if severity is None else 0.05 + 0.12 * severity / 100
    else:
        third = rng.uniform(0, 0.012)
        fifth = rng.uniform(0, 0.010)

    signal += np.sqrt(2) * base_rms * (
        third * np.sin(2 * np.pi * 3 * FREQUENCY * time + rng.uniform(0, 2 * np.pi))
        + fifth * np.sin(2 * np.pi * 5 * FREQUENCY * time + rng.uniform(0, 2 * np.pi))
    )

    if snr_db is None:
        snr_db = rng.uniform(30, 50)
    noise_rms = base_rms / (10 ** (snr_db / 20))
    signal += rng.normal(0, noise_rms, size=N_SAMPLES)
    return time, signal


def cycle_rms(signal):
    samples = np.asarray(signal, dtype=float)
    if samples.shape != (N_SAMPLES,) or not np.all(np.isfinite(samples)):
        raise ValueError(f"Signal must contain {N_SAMPLES} finite samples")
    cycles = samples.reshape(CYCLES, SAMPLES_PER_CYCLE)
    return np.sqrt(np.mean(cycles ** 2, axis=1))


def harmonic_levels(signal):
    spectrum = 2 * np.abs(np.fft.rfft(signal)) / len(signal)
    frequency_resolution = SAMPLE_RATE / N_SAMPLES
    indices = (np.arange(1, 10) * FREQUENCY / frequency_resolution).astype(int)
    peak_levels = spectrum[indices]
    fundamental = max(peak_levels[0], 1e-12)
    ratios = peak_levels / fundamental
    return ratios


def extract_features(signal):
    rms = cycle_rms(signal) / NOMINAL_RMS
    harmonic_ratios = harmonic_levels(signal)
    return np.array([
        np.min(rms),
        np.max(rms),
        np.mean(rms),
        np.std(rms),
        np.max(rms) - np.min(rms),
        100 * harmonic_ratios[2],
        100 * harmonic_ratios[4],
        100 * np.linalg.norm(harmonic_ratios[1:]),
    ])


def train_and_evaluate(samples_per_class=300, seed=2026):
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for label in CLASSES:
        for _ in range(samples_per_class):
            _, waveform = generate_signal(label, rng)
            features.append(extract_features(waveform))
            labels.append(label)

    train_x, test_x, train_y, test_y = train_test_split(
        np.asarray(features),
        np.asarray(labels),
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    matrix = confusion_matrix(test_y, predictions, labels=CLASSES)
    return model, accuracy, matrix, len(train_y), len(test_y)
