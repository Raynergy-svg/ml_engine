"""Phase 75 — DisagreementRecorder + AdaptiveDisagreementFloor tests."""

from __future__ import annotations
import json, tempfile
from pathlib import Path
import pytest

from src.scanner.disagreement_recorder import DisagreementRecorder
from src.scanner.disagreement_gap_analyzer import analyze, CLASS_NEAR_THRESHOLD, CLASS_MODERATE, CLASS_FAR_ABOVE, CLASS_INSUFFICIENT_DATA
from src.scanner.adaptive_disagreement_floor import AdaptiveDisagreementFloor

ENGINE_SRC = Path("src/scanner/engine.py")


def _make_rec(**kw):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        p = Path(f.name)
    p.unlink()
    return DisagreementRecorder(state_path=p, **kw), p


def _make_adf(**kw):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    p.unlink()
    return AdaptiveDisagreementFloor(state_path=p, **kw), p


class TestDisagreementRecorder:
    def test_record_and_distribution(self):
        rec, _ = _make_rec()
        rec.record(pair="USD_CHF", disagreement=0.33, hard_blocked=True)
        rec.record(pair="GBP_USD", disagreement=0.67, hard_blocked=True)
        rec.record(pair="AUD_USD", disagreement=0.0, hard_blocked=False)
        d = rec.get_distribution()
        assert d["count"] == 3
        assert d["mean"] == pytest.approx(0.3333, abs=0.01)

    def test_fail_distribution(self):
        rec, _ = _make_rec()
        rec.record(pair="A", disagreement=0.33, hard_blocked=True)
        rec.record(pair="B", disagreement=0.67, hard_blocked=True)
        rec.record(pair="C", disagreement=0.0, hard_blocked=False)
        fd = rec.get_fail_distribution()
        assert fd["count"] == 2

    def test_pass_rate(self):
        rec, _ = _make_rec()
        rec.record(pair="A", disagreement=0.0, hard_blocked=False)
        rec.record(pair="B", disagreement=0.33, hard_blocked=True)
        assert rec.get_pass_rate() == pytest.approx(0.5)

    def test_save_load_roundtrip(self):
        rec1, path = _make_rec()
        rec1.record(pair="A", disagreement=0.33, hard_blocked=True)
        rec1.save_state()
        rec2 = DisagreementRecorder(state_path=path)
        assert rec2.get_distribution()["count"] == 1
        path.unlink(missing_ok=True)


class TestDisagreementGapAnalyzer:
    def test_near_threshold(self):
        rec, _ = _make_rec()
        for _ in range(5):
            rec.record(pair="A", disagreement=0.33, hard_blocked=True)
        result = analyze(rec, disagreement_hard_floor=0.30)
        assert result["classification"] == CLASS_NEAR_THRESHOLD  # gap=0.03

    def test_moderate(self):
        rec, _ = _make_rec()
        for _ in range(5):
            rec.record(pair="A", disagreement=0.67, hard_blocked=True)
        result = analyze(rec, disagreement_hard_floor=0.30)
        assert result["classification"] == CLASS_FAR_ABOVE  # gap=0.37

    def test_insufficient_data(self):
        rec, _ = _make_rec()
        rec.record(pair="A", disagreement=0.33, hard_blocked=True)
        result = analyze(rec, disagreement_hard_floor=0.30)
        assert result["classification"] == CLASS_INSUFFICIENT_DATA


class TestAdaptiveDisagreementFloor:
    def test_triggers_on_near_threshold_quantization_aware(self):
        """With 5 heuristics, floor=0.30 should jump to 0.41 (just above 0.40 boundary)."""
        adf, _ = _make_adf(cooldown=0, max_adaptations=2, ceiling=0.50)
        result = adf.tick(CLASS_NEAR_THRESHOLD, 0.30)
        # Should jump to 0.40 + 0.01 = 0.41
        assert result is not None
        assert result > 0.40   # above the 2/5 boundary
        assert result < 0.43   # not overshooting

    def test_each_step_crosses_boundary(self):
        """With ceiling=0.70, two adaptations should cross 0.40 and 0.60 boundaries."""
        adf, _ = _make_adf(cooldown=0, max_adaptations=3, ceiling=0.70)
        step1 = adf.tick(CLASS_NEAR_THRESHOLD, 0.30)
        assert step1 is not None
        assert step1 > 0.40   # crossed 2/5 boundary
        assert step1 < 0.43

        step2 = adf.tick(CLASS_NEAR_THRESHOLD, step1)
        assert step2 is not None
        assert step2 > 0.60   # crossed 3/5 boundary
        assert step2 < 0.63

    def test_respects_ceiling(self):
        """Default ceiling=0.50 should block the 0.60 boundary step."""
        adf, _ = _make_adf(cooldown=0, max_adaptations=5, ceiling=0.50)
        step1 = adf.tick(CLASS_NEAR_THRESHOLD, 0.30)  # → ~0.41 (crosses 0.40)
        assert step1 is not None
        step2 = adf.tick(CLASS_NEAR_THRESHOLD, step1)  # → ~0.61 would exceed ceiling 0.50
        assert step2 is None

    def test_respects_max_adaptations(self):
        adf, _ = _make_adf(cooldown=0, max_adaptations=1, ceiling=0.70)
        adf.tick(CLASS_NEAR_THRESHOLD, 0.30)
        assert adf.tick(CLASS_NEAR_THRESHOLD, 0.34) is None

    def test_non_near_returns_none(self):
        adf, _ = _make_adf()
        for cls in ("FAR_ABOVE", "MODERATE", "INSUFFICIENT_DATA"):
            assert adf.tick(cls, 0.30) is None

    def test_no_boundary_above_returns_none(self):
        """If current floor is already above all boundaries, return None."""
        adf, _ = _make_adf(cooldown=0, max_adaptations=5, ceiling=1.5)
        assert adf.tick(CLASS_NEAR_THRESHOLD, 1.01) is None

    def test_status_includes_boundaries(self):
        adf, _ = _make_adf()
        status = adf.get_status()
        assert status["boundaries"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert status["num_heuristics"] == 5

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        path.unlink()
        adf1 = AdaptiveDisagreementFloor(cooldown=0, state_path=path)
        adf1.tick(CLASS_NEAR_THRESHOLD, 0.30)
        adf1.save_state()
        adf2 = AdaptiveDisagreementFloor(state_path=path)
        assert adf2.get_status()["adaptation_count"] == 1
        path.unlink(missing_ok=True)


class TestPhase75WiringGreps:
    def _src(self):
        return ENGINE_SRC.read_text()

    def test_disagreement_recorder_init(self):
        assert "_disagreement_recorder = None" in self._src()

    def test_disagreement_recorder_import(self):
        assert "DisagreementRecorder" in self._src()

    def test_disagreement_recorder_record_call(self):
        assert "_disagreement_recorder.record(" in self._src()

    def test_disagreement_recorder_save_state(self):
        assert "_disagreement_recorder.save_state()" in self._src()

    def test_adaptive_disagreement_floor_init(self):
        assert "_adaptive_disagreement_floor = None" in self._src()

    def test_adaptive_disagreement_floor_import(self):
        assert "AdaptiveDisagreementFloor" in self._src()

    def test_adaptive_disagreement_floor_tick(self):
        assert "_adaptive_disagreement_floor.tick(" in self._src()

    def test_adaptive_disagreement_floor_save_state(self):
        assert "_adaptive_disagreement_floor.save_state()" in self._src()

    def test_disagreement_gap_analyzer_import(self):
        assert "disagreement_gap_analyzer" in self._src()

    def test_config_has_disagreement_hard_floor(self):
        from src.scanner.config import ScannerConfig
        config = ScannerConfig()
        assert hasattr(config, 'disagreement_hard_floor')
        assert config.disagreement_hard_floor == 0.30

    def test_all_modules_importable(self):
        from src.scanner.disagreement_recorder import DisagreementRecorder
        from src.scanner.disagreement_gap_analyzer import analyze
        from src.scanner.adaptive_disagreement_floor import AdaptiveDisagreementFloor
        assert all([DisagreementRecorder, analyze, AdaptiveDisagreementFloor])
