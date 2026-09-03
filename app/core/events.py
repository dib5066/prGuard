"""
In-process pub/sub used to stream live review progress to SSE clients.

The review worker runs in the same event loop as the API (it is started
with ``asyncio.create_task`` from the webhook handler), so a plain
in-memory fan-out is enough: the worker calls :func:`publish` as it moves
through phases, and each connected ``GET /api/reviews/{id}/events`` client
holds a queue registered via :func:`subscribe`.

NOTE: this is single-process only. If PRGuard is ever run with more than
one worker process, this must move to Redis pub/sub (or similar) so events
raised in the worker process reach SSE clients on the API process.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# review_id -> set of subscriber queues
_subscribers: dict[int, set["asyncio.Queue[dict[str, Any]]"]] = {}

_MAX_QUEUE = 500


def publish(review_id: int, event: dict[str, Any]) -> None:
    """Fan ``event`` out to every live subscriber for ``review_id``.

    Never raises — a slow/full client must not break the review pipeline.
    """
    queues = _subscribers.get(review_id)
    if not queues:
        return
    for queue in list(queues):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "SSE queue full for review %s; dropping event %s",
                review_id,
                event.get("type"),
            )


@contextmanager
def subscribe(
    review_id: int,
) -> Iterator["asyncio.Queue[dict[str, Any]]"]:
    """Yield a queue that receives every event published for ``review_id``.

    Usage::

        with subscribe(review_id) as queue:
            event = await queue.get()
    """
    queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue(maxsize=_MAX_QUEUE)
    _subscribers.setdefault(review_id, set()).add(queue)
    try:
        yield queue
    finally:
        subs = _subscribers.get(review_id)
        if subs is not None:
            subs.discard(queue)
            if not subs:
                _subscribers.pop(review_id, None)


def subscriber_count(review_id: int) -> int:
    """Number of clients currently streaming this review (for logging)."""
    return len(_subscribers.get(review_id, ()))
