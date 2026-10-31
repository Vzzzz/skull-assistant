"""core/dispatcher.py around the command stream (T-046 gate): the Redis client must not time out under xread(block),
and a command sent while the dispatcher restarts is handled if fresh, answered `failed` if stale."""
import time

from core import dispatcher
from core.dispatcher import CMD_MAX_AGE_S, stale


def test_redis_client_waits_as_long_as_xread_blocks():
    assert dispatcher.r.connection_pool.connection_kwargs["socket_timeout"] is None


def test_stale_by_stream_id():
    now = time.time()
    fresh = f"{int(1000 * (now - CMD_MAX_AGE_S + 5))}-0"     # sent 5 s inside the limit (the restart takes ~10 s)
    old = f"{int(1000 * (now - CMD_MAX_AGE_S - 1))}-3"
    assert not stale(fresh, now)
    assert stale(old, now)
