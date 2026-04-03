# NLP-AI — Claude Code Context

## Project
NLP-AI: Natural Language Processor for AIs — Lie Detector Prototype

## Stack
- Python 3.10+
- librosa (audio loading, STFT, MFCC, spectral centroid, spectral flux)
- parselmouth (Praat: jitter, shimmer, HNR, formants F1-F4)
- scipy (Butterworth bandpass 8-12Hz, peak detection)
- numpy (FFT core, gradient computation)
- matplotlib (forensic visualization)
- fuzzywuzzy (text emotion lexicon matching)

## Files
- `semantic-emotion.py` — Text-based Valence×Arousal cartesian emotion mapping
- `voice_fft_analyzer.py` — Voice FFT forensic analysis (747 lines, 5 extraction stages)
- `requirements.txt` — Dependencies

## Commands
```bash
pip install -r requirements.txt
python semantic-emotion.py
python voice_fft_analyzer.py <audio.wav>
```

## Architecture
```
Audio → librosa.stft (STFT/FFT) → per-frame extraction:
  ├── Peak Amplitude + Gradient (1st/2nd derivative → consistency score)
  ├── MFCC (13 coefs) + Δ + ΔΔ deltas (vocal tract reconfiguration)
  ├── Spectral Flux (timbral change rate frame-to-frame)
  ├── Spectral Centroid (brightness drift → tension indicator)
  ├── Lippold Microtremor (8-12Hz Butterworth bandpass → RMS energy)
  ├── Praat: jitter, shimmer, HNR (vocal stress triad)
  └── Praat: Formants F1-F4 + dispersion (vocal tract length/tension)
→ Cartesian Mapping (X=Spectral Stability, Y=Vocal Arousal)
→ Classification (CONSISTENT / STRESS_DETECTED / INCONSISTENT)
```

## Key Parameters
- n_fft=2048 (93ms windows at 22050Hz)
- hop_length=512 (23ms temporal resolution)
- gradient_threshold=2.5 (steep curve sensitivity, tunable per speaker)
- Praat features optional (graceful degradation without parselmouth)

## Status
Experimental Phase 1 — in testing with behavioral science researchers.
