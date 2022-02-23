from dataclasses import dataclass
from typing import List

from .account import Account


@dataclass(frozen=True)
class AccountFilter:
    def matches(self, account: Account) -> bool:
        raise NotImplementedError()


@dataclass(frozen=True)
class WhitelistFilter(AccountFilter):
    whitelist: List[Account]

    def matches(self, account: Account) -> bool:
        return account in self.whitelist


@dataclass(frozen=True)
class BlacklistFilter(AccountFilter):
    blacklist: List[Account]

    def matches(self, account: Account) -> bool:
        return account not in self.blacklist
