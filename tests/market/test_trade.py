from tradezoo.market import Account, BuyOrder, SellOrder, Trade


def test_partial_trade():
    buyer = Account(cash_balance=1, stock_balance=1)
    buy_order = BuyOrder(buyer, price=2, volume=2)
    seller = Account(cash_balance=2, stock_balance=2)
    sell_order = SellOrder(seller, price=2, volume=2)
    trade = Trade.from_orders(present_order=buy_order, incoming_order=sell_order)
    assert trade.volume == 0.5
    assert trade.cash_amount == 1
