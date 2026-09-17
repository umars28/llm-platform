"""Where quota state lives.

Per-process counters are correct for exactly one replica. The gateway deployed
with two replicas let a tenant spend twice its budget, because each pod enforced
the limit against its own arithmetic -- and the overspend scales linearly with
the replica count, so the fix is not optional for anything that autoscales.

Two backends behind one interface:

**Memory** is the single-process default, used by tests and by a one-replica
development run. It is honest about what it is: `shared` is False, and the
gateway refuses to start with more than one replica behind it.

**Redis** holds the counters centrally. The operations that matter are done
atomically in Lua rather than as read-modify-write from Python, because the race
this is fixing is precisely two processes reading the same number at the same
time. Doing the check and the increment in separate round trips would reintroduce
the bug in a smaller window and make it much harder to see.

Failure policy is fail-closed on the budget and fail-open on the rate limit.
A budget exists to stop a runaway bill, so losing the store must not silently
lift the cap; a rate limiter exists to protect an upstream, and refusing all
traffic because Redis blinked converts a dependency wobble into an outage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Reservation:
    ok: bool
    reserved_usd: float = 0.0
    reason: str = ""


class QuotaStore(Protocol):
    shared: bool

    def reserve(self, tenant: str, amount_usd: float, limit_usd: float) -> Reservation: ...
    def settle(self, tenant: str, estimated_usd: float, actual_usd: float) -> None: ...
    def release(self, tenant: str, amount_usd: float) -> None: ...
    def take_token(self, tenant: str, rate_per_s: float, burst: float) -> tuple[bool, float]: ...
    def usage(self, tenant: str) -> dict[str, float]: ...
    def healthy(self) -> bool: ...


@dataclass
class MemoryStore:
    """Single-process state. Correct for one replica and no more."""

    shared: bool = False
    reserved: dict[str, float] = field(default_factory=dict)
    settled: dict[str, float] = field(default_factory=dict)
    buckets: dict[str, tuple[float, float]] = field(default_factory=dict)

    def reserve(self, tenant: str, amount_usd: float, limit_usd: float) -> Reservation:
        current = self.reserved.get(tenant, 0.0)
        if current + amount_usd > limit_usd:
            return Reservation(False, current, f"${current:.2f} of ${limit_usd:.2f} committed")
        self.reserved[tenant] = current + amount_usd
        return Reservation(True, self.reserved[tenant])

    def settle(self, tenant: str, estimated_usd: float, actual_usd: float) -> None:
        self.reserved[tenant] = max(0.0, self.reserved.get(tenant, 0.0) - estimated_usd + actual_usd)
        self.settled[tenant] = self.settled.get(tenant, 0.0) + actual_usd

    def release(self, tenant: str, amount_usd: float) -> None:
        self.reserved[tenant] = max(0.0, self.reserved.get(tenant, 0.0) - amount_usd)

    def take_token(self, tenant: str, rate_per_s: float, burst: float) -> tuple[bool, float]:
        now = time.monotonic()
        tokens, updated = self.buckets.get(tenant, (burst, now))
        tokens = min(burst, tokens + (now - updated) * rate_per_s)
        if tokens >= 1.0:
            self.buckets[tenant] = (tokens - 1.0, now)
            return True, 0.0
        self.buckets[tenant] = (tokens, now)
        return False, (1.0 - tokens) / rate_per_s if rate_per_s else 1.0

    def usage(self, tenant: str) -> dict[str, float]:
        return {
            "reserved_usd": self.reserved.get(tenant, 0.0),
            "settled_usd": self.settled.get(tenant, 0.0),
        }

    def healthy(self) -> bool:
        return True


# Check-and-increment in one round trip. Two processes reading the same number
# before either writes is the entire bug being fixed here; splitting this into
# GET then INCRBYFLOAT would reintroduce it in a smaller and less visible window.
RESERVE_LUA = """
local reserved = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
if reserved + amount > limit then
  return {0, tostring(reserved)}
end
local now = redis.call('INCRBYFLOAT', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {1, tostring(now)}
"""

# A token bucket evaluated server-side, for the same reason.
TAKE_TOKEN_LUA = """
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens') or ARGV[2])
local updated = tonumber(redis.call('HGET', KEYS[1], 'updated') or ARGV[3])
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
tokens = math.min(burst, tokens + (now - updated) * rate)
local allowed = 0
local wait = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  wait = (1 - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', KEYS[1], 3600)
return {allowed, tostring(wait)}
"""


class RedisStore:
    """Quota state shared across replicas."""

    shared = True

    def __init__(self, url: str, window_seconds: int = 60 * 60 * 24 * 31, client=None) -> None:
        if client is not None:
            self.client = client
        else:
            import redis

            self.client = redis.Redis.from_url(url, decode_responses=True,
                                               socket_timeout=1.0, socket_connect_timeout=1.0)
        self.window_seconds = window_seconds
        self._reserve = self.client.register_script(RESERVE_LUA)
        self._take = self.client.register_script(TAKE_TOKEN_LUA)

    @staticmethod
    def _reserved_key(tenant: str) -> str:
        return f"gw:reserved:{tenant}"

    @staticmethod
    def _settled_key(tenant: str) -> str:
        return f"gw:settled:{tenant}"

    @staticmethod
    def _bucket_key(tenant: str) -> str:
        return f"gw:bucket:{tenant}"

    def reserve(self, tenant: str, amount_usd: float, limit_usd: float) -> Reservation:
        try:
            allowed, current = self._reserve(
                keys=[self._reserved_key(tenant)],
                args=[amount_usd, limit_usd, self.window_seconds],
            )
        except Exception as exc:
            # Fail closed: a budget that lifts itself when its store is
            # unreachable is not a budget.
            return Reservation(False, 0.0, f"quota store unavailable: {exc}")
        current = float(current)
        if not int(allowed):
            return Reservation(False, current, f"${current:.2f} of ${limit_usd:.2f} committed")
        return Reservation(True, current)

    def settle(self, tenant: str, estimated_usd: float, actual_usd: float) -> None:
        try:
            pipe = self.client.pipeline()
            pipe.incrbyfloat(self._reserved_key(tenant), actual_usd - estimated_usd)
            pipe.incrbyfloat(self._settled_key(tenant), actual_usd)
            pipe.expire(self._settled_key(tenant), self.window_seconds)
            pipe.execute()
        except Exception:
            # A lost settlement leaves the estimate reserved, which errs towards
            # under-spending. Losing the reservation would err towards over-.
            pass

    def release(self, tenant: str, amount_usd: float) -> None:
        try:
            self.client.incrbyfloat(self._reserved_key(tenant), -amount_usd)
        except Exception:
            pass

    def take_token(self, tenant: str, rate_per_s: float, burst: float) -> tuple[bool, float]:
        try:
            allowed, wait = self._take(
                keys=[self._bucket_key(tenant)],
                args=[rate_per_s, burst, time.time()],
            )
        except Exception:
            # Fail open: refusing all traffic because the rate-limit store
            # blinked turns a dependency wobble into an outage.
            return True, 0.0
        return bool(int(allowed)), float(wait)

    def usage(self, tenant: str) -> dict[str, float]:
        try:
            reserved, settled = self.client.mget(
                self._reserved_key(tenant), self._settled_key(tenant)
            )
        except Exception:
            return {"reserved_usd": 0.0, "settled_usd": 0.0}
        return {
            "reserved_usd": float(reserved or 0.0),
            "settled_usd": float(settled or 0.0),
        }

    def healthy(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def build_store(url: str | None, client=None) -> QuotaStore:
    """Redis when configured, memory otherwise."""
    if client is not None or url:
        return RedisStore(url or "", client=client)
    return MemoryStore()
