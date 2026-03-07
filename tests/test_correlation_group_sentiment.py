from src.training.correlation_group_config import (
    get_correlated_direction_map,
    get_pairs_for_baskets,
    get_unique_master_pairs,
)


def test_aud_nzd_short_correlation_map_inverts_aud_quote_pairs():
    mapping = get_correlated_direction_map("AUD_NZD", "short")
    assert mapping["AUD_NZD"] == "short"
    assert mapping["EUR_AUD"] == "long"
    assert mapping["GBP_AUD"] == "long"


def test_aussie_basket_expansion_includes_crosses_and_commodity_pairs():
    pairs = get_pairs_for_baskets(["aussie"])
    assert "AUD_USD" in pairs
    assert "NZD_USD" in pairs
    assert "AUD_NZD" in pairs
    assert "EUR_AUD" in pairs
    assert "GBP_AUD" in pairs


def test_unique_master_pairs_are_deduplicated():
    masters = get_unique_master_pairs()
    assert "EUR_GBP" in masters
    assert "AUD_NZD" in masters
    assert masters.count("EUR_GBP") == 1
