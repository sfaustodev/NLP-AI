"""
NLP-AI Voice FFT Analyzer — Lie Detection Prototype
=====================================================
Forensic frequency analysis using Fast Fourier Transform to detect
consistency changes in vocal amplitude peaks mapped to a cartesian plane.

Hypothesis:
    If the line connecting FFT amplitude peaks over time shows abrupt
    steep curves (high gradient), this correlates with vocal stress
    patterns observed in deceptive speech — increased jitter, shimmer
    perturbations, and HNR (Harmonics-to-Noise Ratio) drops.

Pipeline:
    Voice → FFT (STFT) → Peak Extraction → Gradient Analysis →
    Cartesian Mapping → Consistency Score → Veracity Classification

Author: sfaustodev
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
import json
import os

# ── Optional heavy imports (graceful degradation) ────────────────────────────

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class VocalStressFeatures:
    """Forensic vocal stress indicators extracted from audio."""
    mean_f0: float = 0.0          # Fundamental frequency (Hz)
    std_f0: float = 0.0           # Pitch instability
    jitter: float = 0.0           # Period-to-period frequency perturbation (%)
    shimmer: float = 0.0          # Period-to-period amplitude perturbation (%)
    hnr: float = 0.0              # Harmonics-to-Noise Ratio (dB)
    spectral_centroid: float = 0.0  # Brightness / tension indicator
    peak_gradient_mean: float = 0.0  # Mean slope of amplitude peak line
    peak_gradient_std: float = 0.0   # Gradient inconsistency (KEY metric)
    consistency_score: float = 0.0   # 0-100, lower = more inconsistent = stress
    cartesian_x: float = 0.0        # Mapped valence (consistency axis)
    cartesian_y: float = 0.0        # Mapped arousal (stress intensity axis)


@dataclass
class LieDetectorResult:
    """Final classification result with confidence and trajectory."""
    classification: str = "UNDETERMINED"  # CONSISTENT / STRESS_DETECTED / INCONSISTENT
    confidence: float = 0.0               # 0-100%
    stress_level: str = "UNKNOWN"         # LOW / MODERATE / HIGH / CRITICAL
    features: VocalStressFeatures = field(default_factory=VocalStressFeatures)
    trajectory: list = field(default_factory=list)  # Cartesian trajectory points
    raw_gradients: list = field(default_factory=list)
    spectral: dict = field(default_factory=dict)     # MFCC, flux, microtremor data
    formants: dict = field(default_factory=dict)      # F1-F4 + dispersion


# ── Core FFT Engine ──────────────────────────────────────────────────────────

class VoiceFFTAnalyzer:
    """
    Forensic voice analyzer using STFT peak amplitude consistency.

    The core idea: truthful speech maintains relatively stable amplitude
    peaks across FFT frames. Under cognitive stress (deception), the
    laryngeal muscles tense involuntarily, causing micro-perturbations
    in both frequency (jitter) and amplitude (shimmer) that manifest
    as steep gradient changes in the FFT peak envelope.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 22050,
        f0_min: float = 75.0,
        f0_max: float = 500.0,
        gradient_threshold: float = 2.5,  # Steep curve sensitivity
        consistency_window: int = 10,      # Frames for rolling analysis
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.gradient_threshold = gradient_threshold
        self.consistency_window = consistency_window

        self._check_dependencies()

    def _check_dependencies(self):
        """Check and report available analysis capabilities."""
        self.capabilities = {
            "fft_analysis": HAS_LIBROSA,
            "vocal_stress": HAS_PRAAT,
            "peak_detection": HAS_SCIPY,
        }
        missing = [k for k, v in self.capabilities.items() if not v]
        if missing:
            print(f"[WARN] Missing capabilities: {missing}")
            print("       Install: pip install librosa parselmouth scipy")

    # ── Audio Loading ────────────────────────────────────────────────────

    def load_audio(self, filepath: str) -> tuple:
        """Load audio file, return (signal, sample_rate)."""
        if not HAS_LIBROSA:
            raise ImportError("librosa required: pip install librosa")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        signal, sr = librosa.load(filepath, sr=self.sr, mono=True)
        print(f"[OK] Loaded: {filepath} | {len(signal)/sr:.2f}s | {sr}Hz")
        return signal, sr

    # ── FFT Peak Extraction ──────────────────────────────────────────────

    def extract_fft_peaks(self, signal: np.ndarray) -> dict:
        """
        Apply STFT and extract peak amplitudes per frame.

        Returns dict with:
            - stft_matrix: complex STFT result
            - magnitude: |STFT| magnitude spectrum
            - peak_amplitudes: max amplitude per frame (the "consistency line")
            - peak_frequencies: frequency of max amplitude per frame
            - time_axis: time points for each frame
            - freq_axis: frequency bins
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa required for STFT")

        # Short-Time Fourier Transform
        D = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Frequency and time axes
        freq_axis = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        time_axis = librosa.frames_to_time(
            np.arange(magnitude.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Extract peak amplitude and its frequency for each frame
        # Filter to vocal range only (f0_min to f0_max Hz)
        vocal_mask = (freq_axis >= self.f0_min) & (freq_axis <= self.f0_max * 4)
        vocal_magnitude = magnitude[vocal_mask, :]
        vocal_freqs = freq_axis[vocal_mask]

        peak_indices = np.argmax(vocal_magnitude, axis=0)
        peak_amplitudes = np.array([
            vocal_magnitude[idx, frame]
            for frame, idx in enumerate(peak_indices)
        ])
        peak_frequencies = vocal_freqs[peak_indices]

        return {
            "stft_matrix": D,
            "magnitude": magnitude,
            "peak_amplitudes": peak_amplitudes,
            "peak_frequencies": peak_frequencies,
            "time_axis": time_axis,
            "freq_axis": freq_axis,
        }

    # ── MFCC & Spectral Features ───────────────────────────────────────

    def extract_spectral_features(self, signal: np.ndarray) -> dict:
        """
        Extract MFCC, spectral centroid, spectral flux, and Lippold
        microtremor energy from the audio signal.

        MFCCs capture the spectral envelope — changes in MFCC deltas
        correlate with vocal tract reconfiguration under stress.

        Spectral flux measures frame-to-frame spectral change — high
        flux indicates rapid timbral shifts associated with deception.

        Lippold microtremor (8-12 Hz) is an involuntary muscular
        oscillation that modulates under cognitive load (Lippold, 1970).
        """
        if not HAS_LIBROSA:
            return {}

        # MFCCs (13 coefficients) + deltas
        mfccs = librosa.feature.mfcc(
            y=signal, sr=self.sr, n_mfcc=13,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # Spectral centroid (voice "brightness" — shifts under tension)
        centroid = librosa.feature.spectral_centroid(
            y=signal, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Spectral flux (frame-to-frame spectral change magnitude)
        D = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))

        # Lippold microtremor isolation (8-12 Hz bandpass)
        microtremor_energy = np.array([])
        if HAS_SCIPY:
            from scipy.signal import butter, sosfilt
            nyq = self.sr / 2.0
            low, high = 8.0 / nyq, 12.0 / nyq
            if high < 1.0:  # Only if filter is valid for sample rate
                sos = butter(4, [low, high], btype='band', output='sos')
                filtered = sosfilt(sos, signal)
                # RMS energy in microtremor band per frame
                frame_len = self.hop_length
                n_frames = len(filtered) // frame_len
                microtremor_energy = np.array([
                    np.sqrt(np.mean(
                        filtered[i * frame_len:(i + 1) * frame_len] ** 2
                    ))
                    for i in range(n_frames)
                ])

        return {
            "mfccs": mfccs,
            "mfcc_delta": mfcc_delta,
            "mfcc_delta2": mfcc_delta2,
            "spectral_centroid": centroid,
            "spectral_flux": flux,
            "microtremor_energy": microtremor_energy,
            "mfcc_delta_variance": float(np.var(mfcc_delta)),
            "centroid_std": float(np.std(centroid)),
            "flux_mean": float(np.mean(flux)),
            "microtremor_rms": float(np.sqrt(np.mean(
                microtremor_energy ** 2
            ))) if len(microtremor_energy) > 0 else 0.0,
        }

    # ── Formant Extraction (Praat) ───────────────────────────────────────

    def extract_formants(self, filepath: str) -> dict:
        """
        Extract formant frequencies (F1-F4) via Praat/Parselmouth.

        Formant dispersion correlates with vocal tract length and tension.
        Under stress, formant spacing narrows as the larynx rises and
        the vocal tract shortens involuntarily (Fernandes et al., 2024).
        """
        formants = {"f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0, "dispersion": 0.0}

        if not HAS_PRAAT:
            return formants

        try:
            sound = parselmouth.Sound(filepath)
            formant_obj = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

            for i, key in enumerate(["f1", "f2", "f3", "f4"], start=1):
                formants[key] = call(formant_obj, "Get mean", i, 0, 0, "Hertz")

            # Formant dispersion: average distance between consecutive formants
            vals = [formants[f"f{i}"] for i in range(1, 5) if formants[f"f{i}"] > 0]
            if len(vals) >= 2:
                formants["dispersion"] = np.mean(np.diff(vals))

            print(f"[OK] Formants: F1={formants['f1']:.0f} F2={formants['f2']:.0f} "
                  f"F3={formants['f3']:.0f} F4={formants['f4']:.0f} "
                  f"Disp={formants['dispersion']:.0f}Hz")

        except Exception as e:
            print(f"[WARN] Formant extraction failed: {e}")

        return formants

    # ── Gradient Analysis (The Lie Detector Core) ────────────────────────

    def analyze_peak_consistency(self, peak_amplitudes: np.ndarray) -> dict:
        """
        Analyze the gradient of the peak amplitude line.

        This is THE core hypothesis:
        - Stable gradient → consistent speech → truthful baseline
        - Steep/abrupt gradient changes → stress perturbation → deception signal

        Returns gradient metrics and consistency classification.
        """
        # Normalize peak amplitudes to [0, 1]
        if peak_amplitudes.max() > 0:
            norm_peaks = peak_amplitudes / peak_amplitudes.max()
        else:
            norm_peaks = peak_amplitudes

        # Compute frame-to-frame gradient (first derivative)
        gradient = np.diff(norm_peaks)

        # Compute gradient of gradient (second derivative — acceleration)
        gradient_2nd = np.diff(gradient)

        # Rolling standard deviation of gradient (inconsistency measure)
        if HAS_SCIPY:
            rolling_std = np.array([
                np.std(gradient[max(0, i - self.consistency_window):i + 1])
                for i in range(len(gradient))
            ])
        else:
            rolling_std = np.array([
                np.std(gradient[max(0, i - self.consistency_window):i + 1])
                for i in range(len(gradient))
            ])

        # Detect steep curves (gradient exceeds threshold)
        steep_mask = np.abs(gradient) > (np.std(gradient) * self.gradient_threshold)
        steep_ratio = np.sum(steep_mask) / len(gradient) if len(gradient) > 0 else 0

        # Consistency score v2: percentile-based, calibrated for real speech
        # Real human speech has gradient_std in range 0.03-0.12 typically
        # Lower gradient_std = MORE consistent (smoother) peaks
        # Higher gradient_std = MORE natural variation
        grad_std = np.std(gradient)

        # Natural speech gradient_std baseline: ~0.06-0.10
        # Over-controlled (lie): < 0.06
        # Highly variable (truth/emotion): > 0.08
        # Score 0-100 where 50 = average natural speech
        # Below 30 = suspiciously smooth, Above 70 = very natural/expressive
        consistency_score = max(0, min(100, (grad_std / 0.10) * 50))

        return {
            "gradient": gradient,
            "gradient_2nd": gradient_2nd,
            "rolling_std": rolling_std,
            "steep_mask": steep_mask,
            "steep_ratio": steep_ratio,
            "consistency_score": consistency_score,
            "normalized_peaks": norm_peaks,
        }

    # ── Praat Vocal Stress Features ──────────────────────────────────────

    def extract_vocal_stress(self, filepath: str) -> VocalStressFeatures:
        """
        Extract forensic vocal stress indicators using Praat/Parselmouth.

        Key indicators:
            - Jitter: pitch perturbation (increased under stress)
            - Shimmer: amplitude perturbation (increased under stress)
            - HNR: voice clarity (decreased under stress)
        """
        features = VocalStressFeatures()

        if not HAS_PRAAT:
            print("[WARN] parselmouth not available, skipping Praat features")
            return features

        try:
            sound = parselmouth.Sound(filepath)

            # Pitch (F0)
            pitch = call(sound, "To Pitch (cc)", 0, self.f0_min, 15,
                         'no', 0.03, 0.45, 0.01, 0.35, 0.14, self.f0_max)
            features.mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            features.std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

            # Harmonicity (HNR)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features.hnr = call(harmonicity, "Get mean", 0, 0)

            # Jitter & Shimmer via PointProcess
            point_process = call(sound, "To PointProcess (periodic, cc)",
                                 self.f0_min, self.f0_max)

            features.jitter = call(point_process,
                                   "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features.shimmer = call(
                [sound, point_process],
                "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            print(f"[OK] Praat features: F0={features.mean_f0:.1f}Hz "
                  f"Jitter={features.jitter:.4f} Shimmer={features.shimmer:.4f} "
                  f"HNR={features.hnr:.1f}dB")

        except Exception as e:
            print(f"[ERR] Praat extraction failed: {e}")

        return features

    # ── Cartesian Mapping ────────────────────────────────────────────────

    def map_to_cartesian(
        self,
        consistency: dict,
        stress_features: VocalStressFeatures
    ) -> VocalStressFeatures:
        """
        Map FFT analysis results to the Naturalness × Involuntary Stress plane.

        CALIBRATED from real voice samples (2026-04-02, n=3):
            Truth:  MFCC_Δ=20.7, flux=5.95, jitter=0.020, microtremor=6.6e-5
            Doubt:  MFCC_Δ=19.0, flux=5.63, jitter=0.019, microtremor=4.2e-5
            Lie:    MFCC_Δ=12.3, flux=2.85, jitter=0.015, microtremor=7.7e-5

        Key discovery: lies produce OVER-CONTROLLED voice (low variation)
        but ELEVATED involuntary microtremor (the body leaks).

        X-axis (Naturalness):
            Right (+) = natural vocal variation → truthful, spontaneous
            Left  (-) = over-controlled, artificially stable → deception signal

        Y-axis (Involuntary Stress):
            Up    (+) = elevated microtremor, pitch instability, HNR drop
            Down  (-) = relaxed baseline, stable involuntary signals
        """
        # ── X = Naturalness Score ─────────────────────────────────────
        # Combines multiple indicators of natural vocal expressiveness
        # Over-controlled speech (deception) scores LOW on all of these

        # MFCC delta variance: how much the vocal tract shape changes
        # Truth baseline: ~18-22, Lie baseline: ~10-14
        mfcc_dv = stress_features.spectral_centroid  # stored centroid_std here temporarily
        # Use actual MFCC delta variance if available (passed via spectral dict later)

        # Gradient variation: natural speech has more amplitude peak variation
        grad_std = consistency.get("gradient", np.array([0]))
        grad_variation = float(np.std(grad_std)) if len(grad_std) > 0 else 0

        # Jitter contribution: natural speech has moderate jitter (~0.015-0.025)
        # Very low jitter (<0.012) = suspiciously controlled
        jitter_naturalness = min(stress_features.jitter * 200, 5)  # 0.02 → 4.0

        # Shimmer contribution: similar pattern
        shimmer_naturalness = min(stress_features.shimmer * 30, 5)  # 0.10 → 3.0

        # Consistency score from gradient analysis (higher = more variation = natural)
        consistency_factor = consistency["consistency_score"] / 25.0  # 0-4 range

        naturalness = (
            jitter_naturalness +
            shimmer_naturalness +
            consistency_factor +
            grad_variation * 30
        )
        # Map to [-8, +8] range centered at typical speech
        stress_features.cartesian_x = max(-8, min(8, naturalness - 6))

        # ── Y = Involuntary Stress Score ──────────────────────────────
        # Signals the speaker CANNOT consciously control

        # HNR penalty: lower HNR = voice degradation under stress
        # Normal speech: 15-25 dB, stressed: <15 dB
        hnr_stress = max(0, (18 - stress_features.hnr)) * 0.8

        # Pitch instability: high std_f0 = involuntary pitch jumps
        pitch_stress = min(stress_features.std_f0 / 10, 4)  # 20Hz std → 2.0

        # Steep gradient ratio: involuntary spectral disruptions
        steep = consistency["steep_ratio"]
        steep_stress = steep * 15

        involuntary_stress = hnr_stress + pitch_stress + steep_stress

        # Map to [-2, +8] range
        stress_features.cartesian_y = max(-2, min(8, involuntary_stress - 1))

        # ── Store gradient metrics ────────────────────────────────────
        stress_features.consistency_score = consistency["consistency_score"]
        stress_features.peak_gradient_mean = float(np.mean(np.abs(
            consistency["gradient"]
        )))
        stress_features.peak_gradient_std = float(np.std(consistency["gradient"]))

        return stress_features

    # ── Full Analysis Pipeline ───────────────────────────────────────────

    def analyze(self, filepath: str) -> LieDetectorResult:
        """
        Full forensic analysis pipeline:
            Audio → FFT → Peaks → Gradient → Stress → Cartesian → Classification
        """
        result = LieDetectorResult()

        # 1. Load audio
        signal, sr = self.load_audio(filepath)

        # 2. FFT peak extraction
        fft_data = self.extract_fft_peaks(signal)

        # 3. Gradient consistency analysis
        consistency = self.analyze_peak_consistency(fft_data["peak_amplitudes"])

        # 4. Vocal stress features (Praat)
        stress = self.extract_vocal_stress(filepath)

        # 4b. Spectral features (MFCC, flux, microtremor)
        spectral = self.extract_spectral_features(signal)
        if spectral:
            stress.spectral_centroid = spectral.get("centroid_std", 0.0)

        # 4c. Formant tracking
        formants = self.extract_formants(filepath)

        # 5. Cartesian mapping (now includes spectral + formant data)
        stress = self.map_to_cartesian(consistency, stress)

        # 5b. Refine X-axis with spectral data (MFCC delta var + flux)
        # These are the strongest discriminators from calibration data:
        #   MFCC Δ var: truth=20.7, doubt=19.0, lie=12.3 (40% drop)
        #   Flux:       truth=5.95, doubt=5.63, lie=2.85 (52% drop)
        #   Microtremor: truth=6.6e-5, doubt=4.2e-5, lie=7.7e-5 (17% rise)
        if spectral:
            mfcc_dv = spectral.get("mfcc_delta_variance", 15)
            flux = spectral.get("flux_mean", 4)
            microtremor = spectral.get("microtremor_rms", 0)

            # MFCC delta variance: strongest X discriminator
            # truth~20 → +3, doubt~19 → +2.5, lie~12 → -1.5
            mfcc_boost = max(-5, min(5, (mfcc_dv - 16) / 2))

            # Spectral flux: second strongest X discriminator
            # truth~6 → +2, doubt~5.6 → +1.6, lie~2.8 → -1.2
            flux_boost = max(-4, min(4, (flux - 4) / 1.2))

            # Apply spectral refinement to X (naturalness)
            stress.cartesian_x = max(-8, min(8,
                stress.cartesian_x + mfcc_boost + flux_boost
            ))

            # Microtremor contribution to Y (involuntary stress)
            # The body leaks: truth=6.6e-5, doubt=4.2e-5, lie=7.7e-5
            # Elevated microtremor with LOW naturalness = deception
            if microtremor > 0:
                micro_stress = max(0, (microtremor - 4e-5) / 2e-5) * 1.2
                stress.cartesian_y = max(-2, min(8,
                    stress.cartesian_y + micro_stress
                ))

        # 6. Build trajectory (naturalness vs stress over time)
        norm_peaks = consistency["normalized_peaks"]
        gradient = consistency["gradient"]
        step = max(1, len(norm_peaks) // 20)  # ~20 trajectory points
        trajectory = []
        for i in range(0, len(norm_peaks), step):
            chunk_end = min(i + step, len(gradient))
            if i < len(gradient):
                local_grad_std = np.std(gradient[i:chunk_end])
                local_naturalness = local_grad_std * 30 - 3  # X: more variation = right
                local_stress = max(0, local_grad_std * 15 - 0.5)  # Y: gradient disruptions
                trajectory.append((
                    max(-8, min(8, local_naturalness)),
                    max(-2, min(8, local_stress))
                ))

        # 7. Classify using deception index
        # Core principle: deception = LOW naturalness + HIGH involuntary stress
        x = stress.cartesian_x  # Naturalness: positive=natural, negative=over-controlled
        y = stress.cartesian_y  # Involuntary stress: higher=more stress

        # Deception index: how far into the deception zone
        # x < 1.5 is "barely natural" — suspicious when paired with stress
        deception_index = max(0, 1.5 - x) * max(0, y - 1) / 8

        # Naturalness index: overall vocal expressiveness
        naturalness_index = max(0, x)

        if x >= 4 and y < 3:
            # Strong natural + relaxed → Truthful baseline
            result.classification = "TRUTHFUL"
            result.stress_level = "LOW"
            result.confidence = min(92, 60 + naturalness_index * 4)
        elif x >= 1.5 and y >= 3:
            # Natural + stressed → Truth under pressure (nervous but genuine)
            result.classification = "TRUTHFUL_STRESSED"
            result.stress_level = "MODERATE"
            result.confidence = min(88, 55 + naturalness_index * 4)
        elif x >= 1.5 and y < 3:
            # Natural + relaxed → Truthful
            result.classification = "TRUTHFUL"
            result.stress_level = "LOW"
            result.confidence = min(90, 60 + naturalness_index * 4)
        elif x < 1.5 and y >= 2 and deception_index > 0.3:
            # Over-controlled + involuntary stress → DECEPTION
            result.classification = "DECEPTION_DETECTED"
            result.stress_level = "HIGH"
            result.confidence = min(88, 45 + deception_index * 15)
        elif x < 1.5 and y < 2:
            # Over-controlled + relaxed → Rehearsed/Scripted
            result.classification = "REHEARSED"
            result.stress_level = "LOW"
            result.confidence = min(72, 40 + abs(1.5 - x) * 5)
        else:
            # Borderline
            result.classification = "UNCERTAIN"
            result.stress_level = "MODERATE"
            result.confidence = max(35, 50 - deception_index * 5)

        result.features = stress
        result.trajectory = trajectory
        result.raw_gradients = gradient.tolist()
        result.spectral = spectral if spectral else {}
        result.formants = formants

        return result

    # ── Visualization ────────────────────────────────────────────────────

    def plot_analysis(
        self,
        filepath: str,
        result: LieDetectorResult,
        save_path: str = "lie_detector_plot.png"
    ):
        """
        Generate forensic analysis visualization:
            1. Spectrogram with peak amplitude line
            2. Gradient consistency over time
            3. Cartesian veracity map with trajectory
        """
        signal, sr = self.load_audio(filepath)
        fft_data = self.extract_fft_peaks(signal)
        consistency = self.analyze_peak_consistency(fft_data["peak_amplitudes"])

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            f"NLP-AI — Forensic Voice FFT Analysis\n"
            f"Classification: {result.classification} | "
            f"Stress: {result.stress_level} | "
            f"Confidence: {result.confidence:.1f}%",
            fontsize=16, fontweight='bold', y=0.98
        )

        # ── 1. Spectrogram ───────────────────────────────────────────────
        ax1 = axes[0][0]
        S_db = librosa.amplitude_to_db(fft_data["magnitude"], ref=np.max)
        librosa.display.specshow(
            S_db, sr=sr, hop_length=self.hop_length,
            x_axis='time', y_axis='log', ax=ax1, cmap='magma'
        )
        ax1.set_title("Spectrogram (STFT Magnitude)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Frequency (Hz)")

        # ── 2. Peak Amplitude Line (The Consistency Line) ────────────────
        ax2 = axes[0][1]
        time_ax = fft_data["time_axis"]
        norm_peaks = consistency["normalized_peaks"]
        ax2.plot(time_ax, norm_peaks, color='cyan', linewidth=1.5, alpha=0.7, label='Peak Amplitude')

        # Overlay steep gradient markers
        steep = consistency["steep_mask"]
        steep_times = time_ax[1:][steep]  # gradient is 1 shorter than peaks
        steep_vals = norm_peaks[1:][steep]
        ax2.scatter(steep_times, steep_vals, color='red', s=30, zorder=5,
                    label=f'Steep Gradient ({np.sum(steep)} frames)')

        # Rolling std overlay
        rolling = consistency["rolling_std"]
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(time_ax[1:], rolling, alpha=0.2, color='orange',
                              label='Gradient Instability')
        ax2_twin.set_ylabel("Instability", color='orange', fontsize=10)

        ax2.set_title("Peak Amplitude Consistency Line", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Normalized Amplitude")
        ax2.legend(loc='upper left', fontsize=9)

        # ── 3. Gradient (1st derivative) ─────────────────────────────────
        ax3 = axes[1][0]
        gradient = consistency["gradient"]
        colors = ['red' if s else 'cyan' for s in steep]
        ax3.bar(time_ax[1:], gradient, width=time_ax[1] - time_ax[0] if len(time_ax) > 1 else 0.01,
                color=colors, alpha=0.7)
        ax3.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax3.axhline(np.std(gradient) * self.gradient_threshold, color='red',
                     linewidth=1, linestyle=':', label='Steep threshold')
        ax3.axhline(-np.std(gradient) * self.gradient_threshold, color='red',
                     linewidth=1, linestyle=':')
        ax3.set_title("Frame-to-Frame Gradient (Δ Amplitude)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Gradient")
        ax3.legend(fontsize=9)

        # ── 4. Cartesian Veracity Map ────────────────────────────────────
        ax4 = axes[1][1]
        if result.trajectory:
            traj_x = [p[0] for p in result.trajectory]
            traj_y = [p[1] for p in result.trajectory]
            ax4.plot(traj_x, traj_y, 'o-', color='cyan', linewidth=3,
                     markersize=8, alpha=0.8, label='Voice Trajectory')
            ax4.scatter(traj_x[0], traj_y[0], color='lime', s=200,
                       zorder=5, label='Start')
            ax4.scatter(traj_x[-1], traj_y[-1], color='red', s=300,
                       marker='*', zorder=5, label='End')

        # Final position
        ax4.scatter(result.features.cartesian_x, result.features.cartesian_y,
                   color='yellow', s=400, marker='D', zorder=6,
                   label=f'Final: ({result.features.cartesian_x:.1f}, '
                         f'{result.features.cartesian_y:.1f})')

        # Quadrant labels
        ax4.axhline(0, color='gray', linewidth=1.5, linestyle='--')
        ax4.axvline(0, color='gray', linewidth=1.5, linestyle='--')

        lim = 12
        ax4.set_xlim(-lim, lim)
        ax4.set_ylim(-2, lim)

        q = lim * 0.65
        ax4.text(q, q, 'TRUTHFUL\nUNDER PRESSURE', ha='center', va='center',
                fontsize=13, fontweight='bold',
                bbox=dict(facecolor='lightgreen', alpha=0.7))
        ax4.text(-q, q, 'DECEPTION\nDETECTED', ha='center', va='center',
                fontsize=13, fontweight='bold',
                bbox=dict(facecolor='lightcoral', alpha=0.7))
        ax4.text(q, 1, 'TRUTHFUL\nBASELINE', ha='center', va='center',
                fontsize=13, fontweight='bold',
                bbox=dict(facecolor='lightblue', alpha=0.7))
        ax4.text(-q, 1, 'REHEARSED\nSCRIPTED', ha='center', va='center',
                fontsize=13, fontweight='bold',
                bbox=dict(facecolor='lightyellow', alpha=0.7))

        ax4.set_xlabel('← Over-Controlled ——— Naturalness ——— Natural →', fontsize=12)
        ax4.set_ylabel('← Relaxed ——— Involuntary Stress ——— Stressed →', fontsize=12)
        ax4.set_title("Cartesian Veracity Map", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        for ax in axes.flat:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        fig.suptitle(fig._suptitle.get_text(), color='white', fontsize=16,
                     fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[OK] Plot saved: {save_path}")
        plt.close()

    def to_json(self, result: LieDetectorResult, spectral: dict = None, formants: dict = None) -> str:
        """Export result as JSON for API consumption."""
        data = {
            "classification": result.classification,
            "confidence": round(result.confidence, 2),
            "stress_level": result.stress_level,
            "features": {
                "mean_f0_hz": round(result.features.mean_f0, 2),
                "std_f0_hz": round(result.features.std_f0, 2),
                "jitter": round(result.features.jitter, 6),
                "shimmer": round(result.features.shimmer, 6),
                "hnr_db": round(result.features.hnr, 2),
                "consistency_score": round(result.features.consistency_score, 2),
                "peak_gradient_mean": round(result.features.peak_gradient_mean, 6),
                "peak_gradient_std": round(result.features.peak_gradient_std, 6),
                "cartesian_x": round(result.features.cartesian_x, 2),
                "cartesian_y": round(result.features.cartesian_y, 2),
            },
            "trajectory_points": len(result.trajectory),
        }
        if spectral:
            data["spectral"] = {
                "mfcc_delta_variance": round(spectral.get("mfcc_delta_variance", 0), 6),
                "spectral_centroid_std": round(spectral.get("centroid_std", 0), 2),
                "spectral_flux_mean": round(spectral.get("flux_mean", 0), 4),
                "microtremor_rms_8_12hz": round(spectral.get("microtremor_rms", 0), 8),
            }
        if formants:
            data["formants"] = {
                "f1_hz": round(formants.get("f1", 0), 1),
                "f2_hz": round(formants.get("f2", 0), 1),
                "f3_hz": round(formants.get("f3", 0), 1),
                "f4_hz": round(formants.get("f4", 0), 1),
                "dispersion_hz": round(formants.get("dispersion", 0), 1),
            }
        return json.dumps(data, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)


# ── CLI Usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 60)
        print("NLP-AI Voice FFT Analyzer — Lie Detection Prototype")
        print("=" * 60)
        print()
        print("Usage: python voice_fft_analyzer.py <audio_file.wav>")
        print()
        print("Dependencies:")
        print(f"  librosa     : {'OK' if HAS_LIBROSA else 'MISSING (pip install librosa)'}")
        print(f"  parselmouth : {'OK' if HAS_PRAAT else 'MISSING (pip install praat-parselmouth)'}")
        print(f"  scipy       : {'OK' if HAS_SCIPY else 'MISSING (pip install scipy)'}")
        print()
        print("Example:")
        print("  python voice_fft_analyzer.py sample_truth.wav")
        print("  python voice_fft_analyzer.py sample_lie.wav")
        sys.exit(0)

    filepath = sys.argv[1]
    analyzer = VoiceFFTAnalyzer()
    result = analyzer.analyze(filepath)

    print()
    print("=" * 60)
    print("FORENSIC VOICE ANALYSIS RESULT")
    print("=" * 60)
    print(analyzer.to_json(result, spectral=result.spectral, formants=result.formants))
    print("=" * 60)

    # Generate plot
    plot_path = os.path.splitext(filepath)[0] + "_analysis.png"
    analyzer.plot_analysis(filepath, result, save_path=plot_path)
