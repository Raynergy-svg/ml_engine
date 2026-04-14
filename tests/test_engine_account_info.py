from src.scanner.engine import Scanner


def test_get_account_info_exposes_both_oanda_and_normalized_keys():
    scanner = Scanner.__new__(Scanner)
    scanner._oanda = type(
        "MockOanda",
        (),
        {
            "get_account_summary": staticmethod(lambda: {
                "balance": 100000.0,
                "NAV": 101861.04,
                "unrealizedPL": 123.45,
                "marginUsed": 4567.89,
                "marginAvailable": 97293.15,
                "openPositionCount": 2,
                "openTradeCount": 3,
            })
        },
    )()
    scanner._init_oanda_client = lambda: True

    info = scanner.get_account_info()

    assert info["NAV"] == 101861.04
    assert info["nav"] == 101861.04
    assert info["unrealizedPL"] == 123.45
    assert info["unrealized_pl"] == 123.45
    assert info["openTradeCount"] == 3
    assert info["open_trades"] == 3
