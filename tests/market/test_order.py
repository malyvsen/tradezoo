from tradezoo.market import Account, BuyOrder, SellOrder


def test_exact_match():
    buyer = Account(cash_balance=1, stock_balance=1)
    buy_order = BuyOrder(buyer, price=0.5, volume=1)
    seller = Account(cash_balance=2, stock_balance=2)
    sell_order = SellOrder(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)


def test_price_mismatch():
    buyer = Account(cash_balance=1, stock_balance=1)
    buy_order = BuyOrder(buyer, price=0.4, volume=1)
    seller = Account(cash_balance=2, stock_balance=2)
    sell_order = SellOrder(seller, price=0.6, volume=1)
    assert not buy_order.matches(sell_order)
    assert not sell_order.matches(buy_order)


def test_crossed_price():
    buyer = Account(cash_balance=1, stock_balance=1)
    buy_order = BuyOrder(buyer, price=1, volume=1)
    seller = Account(cash_balance=2, stock_balance=2)
    sell_order = SellOrder(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)
