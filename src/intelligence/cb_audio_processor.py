"""Central Bank Audio Tone Processor.

US-004: Extracts sentiment, tone, and hesitation metrics from central bank
press conference audio. Supports Powell (Fed), Ueda (BoJ), Bailey (BoE) pressers.

Optional dependencies:
  - whisper (OpenAI Whisper transcription)
  - librosa (audio feature extraction)
  - soundfile (audio I/O)

Graceful fallback: if audio libs are missing, the processor returns a neutral
score with a degradation flag.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False
    whisper = None  # type: ignore[assignment]

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False
    librosa = None  # type: ignore[assignment]

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False
    sf = None  # type: ignore[assignment]


# ------------------------------------------------------------------
# Result data structure
# ------------------------------------------------------------------

@dataclass
class ToneMetrics:
    """Metrics extracted from central bank press conference audio."""

    speaker: str
    transcript: str = ""
    speaking_rate_wpm: float = 0.0
    pause_frequency: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_variance: float = 0.0
    filler_word_count: int = 0
    false_start_count: int = 0
    hesitation_score: float = 0.0
    dovish_hawkish_score: float = 0.0  # -1 = dovish, +1 = hawkish
    confidence: float = 0.0
    degraded: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "transcript": self.transcript,
            "speaking_rate_wpm": self.speaking_rate_wpm,
            "pause_frequency": self.pause_frequency,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_variance": self.pitch_variance,
            "filler_word_count": self.filler_word_count,
            "false_start_count": self.false_start_count,
            "hesitation_score": self.hesitation_score,
            "dovish_hawkish_score": self.dovish_hawkish_score,
            "confidence": self.confidence,
            "degraded": self.degraded,
            "reason": self.reason,
        }


# ------------------------------------------------------------------
# Keyword lexicons
# ------------------------------------------------------------------

HAWKISH_WORDS = frozenset([
    "tighten", "tightening", "raise", "hike", "inflation", "elevated",
    "above target", "restrictive", "overshoot", "hot", "pressure",
    "wage pressure", "demand", "strong", "robust", "accelerate",
    "forceful", "decisive", "front-load",
])

DOVISH_WORDS = frozenset([
    "ease", "easing", "cut", "lower", "pause", "hold", "stable",
    "below target", "disinflation", "cooling", "soften", "soft",
    "weak", "fragile", "uncertainty", "cautious", "patient",
    "data-dependent", "gradual", "moderate",
])

FILLER_WORDS = frozenset([
    "um", "uh", "ah", "er", "like", "you know", "i mean", "sort of",
    "kind of", "basically", "literally", "actually",
])


# ------------------------------------------------------------------
# Audio processor
# ------------------------------------------------------------------

class CBAudioProcessor:
    """Process central bank press conference audio and extract tone metrics."""

    def __init__(self, whisper_model: str = "base"):
        self.whisper_model_name = whisper_model
        self._whisper = None
        if WHISPER_AVAILABLE:
            try:
                self._whisper = whisper.load_model(whisper_model)  # type: ignore[attr-defined]
                logger.info("CBAudioProcessor: Whisper model '%s' loaded", whisper_model)
            except Exception as exc:
                logger.warning("CBAudioProcessor: failed to load Whisper: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        audio_path: Optional[Path] = None,
        audio_url: Optional[str] = None,
        speaker: str = "unknown",
        transcript_text: Optional[str] = None,
    ) -> ToneMetrics:
        """Process audio (or pre-supplied transcript) and return tone metrics.

        Args:
            audio_path: Local path to audio file (wav/mp3).
            audio_url: URL to audio stream (not yet implemented — falls back).
            speaker: Name of central bank speaker (e.g., "powell", "ueda").
            transcript_text: Optional pre-transcribed text (skip Whisper).
        """
        if audio_url and not audio_path:
            logger.warning("CBAudioProcessor: URL download not yet implemented; returning degraded.")
            return self._degraded_result(speaker, "url_download_not_implemented")

        # If transcript is provided, skip audio processing
        if transcript_text is not None:
            if not transcript_text.strip():
                return self._degraded_result(speaker, "transcription_failed")
            text_metrics = self._analyze_text(transcript_text)
            return self._combine_metrics(
                speaker, transcript_text, text_metrics,
                {"speaking_rate_wpm": 0.0, "pause_frequency": 0.0,
                 "pitch_mean": 0.0, "pitch_std": 0.0, "pitch_variance": 0.0}
            )

        if not audio_path:
            return self._degraded_result(speaker, "no_audio_or_transcript")

        if not audio_path.exists():
            return self._degraded_result(speaker, f"audio_not_found:{audio_path}")

        # Step 1: Transcribe
        transcript = self._transcribe(audio_path)
        if not transcript:
            return self._degraded_result(speaker, "transcription_failed")

        # Step 2: Audio features
        audio_features = self._extract_audio_features(audio_path)

        # Step 3: Text analysis
        text_metrics = self._analyze_text(transcript)

        # Step 4: Combine
        return self._combine_metrics(speaker, transcript, text_metrics, audio_features)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------
    def _transcribe(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if self._whisper is None or not WHISPER_AVAILABLE:
            logger.debug("CBAudioProcessor: Whisper unavailable, skipping transcription")
            return None
        try:
            result = self._whisper.transcribe(str(audio_path))
            return result.get("text", "").strip() if result else None  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("CBAudioProcessor: transcription error: %s", exc)
            return None

    def _extract_audio_features(self, audio_path: Path) -> Dict[str, float]:
        """Extract pitch, speaking rate, and pause features from audio."""
        features: Dict[str, float] = {
            "speaking_rate_wpm": 0.0,
            "pause_frequency": 0.0,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_variance": 0.0,
        }
        if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
            logger.debug("CBAudioProcessor: librosa/soundfile unavailable, skipping audio features")
            return features
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)  # type: ignore[attr-defined]
            duration_min = len(y) / sr / 60.0
            if duration_min <= 0:
                return features

            # Pitch (fundamental frequency)
            f0, voiced_flag, _ = librosa.pyin(  # type: ignore[attr-defined]
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")  # type: ignore[attr-defined]
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            if len(voiced_f0) > 0:
                features["pitch_mean"] = float(np.nanmean(voiced_f0))
                features["pitch_std"] = float(np.nanstd(voiced_f0))
                features["pitch_variance"] = features["pitch_std"] ** 2

            # Speaking rate proxy: total frames / duration
            # (crude — better would be syllable count, but this is a heuristic)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # type: ignore[attr-defined]
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)  # type: ignore[attr-defined]
            if duration_min > 0:
                features["speaking_rate_wpm"] = float(len(onsets)) / duration_min

            # Pause frequency: ratio of unvoiced regions
            if voiced_flag is not None:
                voiced_ratio = np.mean(voiced_flag)
                features["pause_frequency"] = float(1.0 - voiced_ratio)

        except Exception as exc:
            logger.warning("CBAudioProcessor: audio feature extraction failed: %s", exc)
        return features

    def _analyze_text(self, transcript: str) -> Dict[str, Any]:
        """Extract text-based tone metrics from transcript."""
        text_lower = transcript.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_count = len(words)

        # Filler words
        filler_count = sum(1 for w in words if w in FILLER_WORDS)

        # False starts (repeated words or self-corrections)
        false_starts = len(re.findall(r"\b(\w+)\s+\1\b", text_lower))
        false_starts += len(re.findall(r"\bno\s+,\s+i\s+mean\b|\bwhat\s+i\s+mean\s+is\b", text_lower))

        # Hawkish / dovish scoring
        hawkish_hits = sum(1 for phrase in HAWKISH_WORDS if phrase in text_lower)
        dovish_hits = sum(1 for phrase in DOVISH_WORDS if phrase in text_lower)
        total_hits = hawkish_hits + dovish_hits
        if total_hits > 0:
            raw_score = (hawkish_hits - dovish_hits) / total_hits
            dovish_hawkish = np.clip(raw_score, -1.0, 1.0)
        else:
            dovish_hawkish = 0.0

        # Hesitation score (0–1)
        hesitation = 0.0
        if word_count > 0:
            hesitation = min((filler_count + false_starts) / max(word_count * 0.05, 1.0), 1.0)

        return {
            "filler_word_count": filler_count,
            "false_start_count": false_starts,
            "hesitation_score": hesitation,
            "dovish_hawkish_score": dovish_hawkish,
            "word_count": word_count,
            "hawkish_hits": hawkish_hits,
            "dovish_hits": dovish_hits,
        }

    def _combine_metrics(
        self,
        speaker: str,
        transcript: str,
        text_metrics: Dict[str, Any],
        audio_features: Dict[str, float],
    ) -> ToneMetrics:
        """Combine audio and text metrics into a final ToneMetrics result."""
        # Confidence: higher when we have both audio features and transcript
        has_audio = any(v > 0 for v in audio_features.values())
        confidence = 0.6 if transcript else 0.0
        if has_audio:
            confidence = 0.85

        return ToneMetrics(
            speaker=speaker,
            transcript=transcript,
            speaking_rate_wpm=audio_features.get("speaking_rate_wpm", 0.0),
            pause_frequency=audio_features.get("pause_frequency", 0.0),
            pitch_mean=audio_features.get("pitch_mean", 0.0),
            pitch_std=audio_features.get("pitch_std", 0.0),
            pitch_variance=audio_features.get("pitch_variance", 0.0),
            filler_word_count=text_metrics.get("filler_word_count", 0),
            false_start_count=text_metrics.get("false_start_count", 0),
            hesitation_score=text_metrics.get("hesitation_score", 0.0),
            dovish_hawkish_score=text_metrics.get("dovish_hawkish_score", 0.0),
            confidence=confidence,
            degraded=False,
            reason="ok",
        )

    def _degraded_result(self, speaker: str, reason: str) -> ToneMetrics:
        return ToneMetrics(
            speaker=speaker,
            degraded=True,
            reason=reason,
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Convenience presets
    # ------------------------------------------------------------------
    @classmethod
    def from_transcript(cls, transcript: str, speaker: str = "unknown") -> ToneMetrics:
        """Process a pre-existing transcript without audio."""
        processor = cls()
        return processor.process(speaker=speaker, transcript_text=transcript)
