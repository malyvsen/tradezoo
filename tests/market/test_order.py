from tradezoo.market import (
    Account,
    BlacklistFilter,
    BuyOrder,
    SellOrder,
    WhitelistFilter,
)


def test_exact_match():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder.public(buyer, price=0.5, volume=1)
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)


def test_price_mismatch():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder.public(buyer, price=0.4, volume=1)
    sell_order = SellOrder.public(seller, price=0.6, volume=1)
    assert not buy_order.matches(sell_order)
    assert not sell_order.matches(buy_order)


def test_crossed_price():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder.public(buyer, price=1, volume=1)
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)


def test_self_trade():
    account = Account(cash_balance=5, asset_balance=5)
    buy_order = BuyOrder.public(account, price=0.5, volume=1)
    sell_order = SellOrder.public(account, price=0.5, volume=1)
    assert not buy_order.matches(sell_order)
    assert not sell_order.matches(buy_order)


def test_whitelist():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder(
        buyer, visibility=WhitelistFilter([buyer]), price=0.5, volume=1
    )
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert not buy_order.matches(sell_order)
    assert not sell_order.matches(buy_order)


def test_whitelist_positive():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder(
        buyer, visibility=WhitelistFilter([seller]), price=0.5, volume=1
    )
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)


def test_blacklist():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder(
        buyer, visibility=BlacklistFilter([seller]), price=0.5, volume=1
    )
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert not buy_order.matches(sell_order)
    assert not sell_order.matches(buy_order)


def test_blacklist_positive():
    buyer = Account(cash_balance=1, asset_balance=1)
    seller = Account(cash_balance=2, asset_balance=2)
    buy_order = BuyOrder(
        buyer, visibility=BlacklistFilter([buyer]), price=0.5, volume=1
    )
    sell_order = SellOrder.public(seller, price=0.5, volume=1)
    assert buy_order.matches(sell_order)
    assert sell_order.matches(buy_order)
