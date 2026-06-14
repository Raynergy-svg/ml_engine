"""Tests for Central Bank Audio Tone Processor (US-004)."""

from __future__ import annotations

import pytest

from src.intelligence.cb_audio_processor import (
    CBAudioProcessor,
    ToneMetrics,
)


class TestToneMetrics:

    def test_to_dict(self):
        t = ToneMetrics(
            speaker="powell",
            transcript="Inflation is elevated.",
            dovish_hawkish_score=0.5,
            confidence=0.8,
        )
        d = t.to_dict()
        assert d["speaker"] == "powell"
        assert d["dovish_hawkish_score"] == 0.5
        assert d["degraded"] is False


class TestCBAudioProcessor:

    def test_init(self):
        p = CBAudioProcessor()
        assert p.whisper_model_name == "base"

    def test_process_no_input(self):
        p = CBAudioProcessor()
        result = p.process(speaker="powell")
        assert result.degraded is True
        assert result.reason == "no_audio_or_transcript"

    def test_process_from_transcript(self):
        p = CBAudioProcessor()
        transcript = (
            "The committee decided to raise the target range for the federal funds rate "
            "by 25 basis points. Inflation remains elevated and well above our long-run goal. "
            "We expect that ongoing increases in the target range will be appropriate."
        )
        result = p.process(speaker="powell", transcript_text=transcript)
        assert result.degraded is False
        assert result.transcript == transcript
        assert result.dovish_hawkish_score > 0  # hawkish words dominate
        assert result.confidence > 0

    def test_process_from_transcript_dovish(self):
        p = CBAudioProcessor()
        transcript = (
            "The economy is cooling and we see signs of disinflation. "
            "We will be patient and data-dependent. There is uncertainty ahead."
        )
        result = p.process(speaker="powell", transcript_text=transcript)
        assert result.degraded is False
        assert result.dovish_hawkish_score < 0  # dovish words dominate

    def test_process_from_transcript_filler_words(self):
        p = CBAudioProcessor()
        transcript = (
            "Um, we are, uh, looking at the data. You know, inflation is, um, "
            "still a concern. I mean, we need to be cautious."
        )
        result = p.process(speaker="powell", transcript_text=transcript)
        assert result.degraded is False
        assert result.filler_word_count > 0
        assert result.hesitation_score > 0

    def test_process_url_fallback(self):
        p = CBAudioProcessor()
        result = p.process(audio_url="https://example.com/audio.mp3", speaker="ueda")
        assert result.degraded is True
        assert "url_download" in result.reason

    def test_classmethod_from_transcript(self):
        result = CBAudioProcessor.from_transcript(
            "We will hold rates steady.", speaker="bailey"
        )
        assert result.degraded is False
        assert result.speaker == "bailey"

    def test_empty_transcript(self):
        p = CBAudioProcessor()
        result = p.process(speaker="powell", transcript_text="")
        assert result.degraded is True
        assert result.reason == "transcription_failed"
