# Electrical Power Quality Monitor

Laptop-only simulation and Random Forest classification of 50 Hz, 230 V voltage signals: normal, sag, swell, and harmonic distortion.

## Run on Windows

Use Python 3.12 or 3.13.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select a voltage condition, disturbance severity, and noise level in the sidebar.

The model is trained on synthetic waveforms and evaluated on held-out synthetic waveforms. Real-world performance is unverified.
