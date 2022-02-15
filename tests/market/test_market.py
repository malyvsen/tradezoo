from tradezoo.market import Market, Account, BuyOrder, SellOrder


def test_exact_match():
    market = Market.from_accounts(
        accounts=[
            Account(cash_balance=1, asset_balance=1),
            Account(cash_balance=2, asset_balance=2),
        ],
    )
    market.submit_(BuyOrder(market.accounts[0], price=0.5, volume=1))
    market.submit_(SellOrder(market.accounts[1], price=0.5, volume=1))
    assert market.orders == []
    assert market.accounts[0] == Account(cash_balance=0.5, asset_balance=2)
    assert market.accounts[1] == Account(cash_balance=2.5, asset_balance=1)


def test_price_mismatch():
    market = Market.from_accounts(
        accounts=[
            Account(cash_balance=5, asset_balance=5),
            Account(cash_balance=6, asset_balance=6),
        ],
    )
    market.submit_(BuyOrder(market.accounts[0], price=2, volume=1))
    market.submit_(SellOrder(market.accounts[1], price=3, volume=1))
    assert len(market.orders) == 2
    assert market.accounts[0] == Account(cash_balance=5, asset_balance=5)
    assert market.accounts[1] == Account(cash_balance=6, asset_balance=6)


def test_crossed_price():
    market = Market.from_accounts(
        accounts=[
            Account(cash_balance=5, asset_balance=5),
            Account(cash_balance=6, asset_balance=6),
        ],
    )
    market.submit_(BuyOrder(market.accounts[0], price=3, volume=1))
    market.submit_(SellOrder(market.accounts[1], price=2, volume=1))
    assert market.orders == []
    assert market.accounts[0] == Account(cash_balance=3, asset_balance=6)
    assert market.accounts[1] == Account(cash_balance=8, asset_balance=5)


def test_multiple_matches():
    market = Market.from_accounts(
        accounts=[
            Account(cash_balance=5, asset_balance=5),
            Account(cash_balance=6, asset_balance=6),
        ],
    )
    market.submit_(BuyOrder(market.accounts[0], price=1, volume=1))
    market.submit_(BuyOrder(market.accounts[0], price=2, volume=1))
    market.submit_(BuyOrder(market.accounts[0], price=3, volume=1))
    market.submit_(SellOrder(market.accounts[1], price=2, volume=1.5))
    assert market.orders == [
        BuyOrder(market.accounts[0], price=1, volume=1),
        BuyOrder(market.accounts[0], price=2, volume=0.5),
    ]
    assert market.accounts[0] == Account(cash_balance=2, asset_balance=6.5)
    assert market.accounts[1] == Account(cash_balance=9, asset_balance=4.5)


def test_exhausting_orderbook():
    market = Market.from_accounts(
        accounts=[
            Account(cash_balance=5, asset_balance=5),
            Account(cash_balance=17, asset_balance=8),
        ],
    )
    market.submit_(SellOrder(market.accounts[0], price=3, volume=1))
    market.submit_(SellOrder(market.accounts[0], price=1, volume=1))
    market.submit_(SellOrder(market.accounts[0], price=2, volume=1))
    market.submit_(BuyOrder(market.accounts[1], price=4, volume=8))
    assert market.orders == [BuyOrder(market.accounts[1], price=4, volume=5)]
    assert market.accounts[0] == Account(cash_balance=17, asset_balance=2)
    assert market.accounts[1] == Account(cash_balance=5, asset_balance=11)
