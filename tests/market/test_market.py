from tradezoo.market import Market, Account, BuyOrder, SellOrder


def test_exact_match():
    market = Market(
        accounts=[
            Account(cash_balance=1, stock_balance=1),
            Account(cash_balance=2, stock_balance=2),
        ],
        orders=[],
    )
    market.submit_(BuyOrder(market.accounts[0], price=0.5, volume=1))
    market.submit_(SellOrder(market.accounts[1], price=0.5, volume=1))
    assert market.orders == []
    assert market.accounts[0] == Account(cash_balance=0.5, stock_balance=2)
    assert market.accounts[1] == Account(cash_balance=2.5, stock_balance=1)
