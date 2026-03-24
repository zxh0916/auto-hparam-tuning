"""eta.py — Compute an ETA timestamp from a remaining-time string.

CLI usage
---------
    python scripts/eta.py [--cron | --iso8601] <remaining>

    <remaining> is a compact duration string composed of one or more
    ``<number><unit>`` tokens (case-insensitive, whitespace optional):

        1h30m        → 1 hour 30 minutes
        90m          → 90 minutes
        3600s        → 3600 seconds
        2h           → 2 hours
        1h 30m 45s   → 1 hour 30 minutes 45 seconds
        0.5h         → 30 minutes

    --iso8601  (default) Print a UTC ISO 8601 timestamp, e.g. ``2026-03-16T18:45:00Z``.
    --cron     Print a one-shot 5-field cron expression in local time, e.g. ``15 20 16 3 *``.

Python usage
------------
    from eta import eta_iso, duration_to_iso8601, duration_to_cron
    from datetime import timedelta

    print(eta_iso(timedelta(hours=1, minutes=30)))
    # → '2026-03-16T18:45:00Z'

    print(duration_to_iso8601("1h30m"))
    # → '2026-03-16T18:45:00Z'

    print(duration_to_cron("1h30m"))
    # → '15 20 16 3 *'  (one-shot cron in local time, now + 1h30m)
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone


def eta_iso(remaining: timedelta) -> str:
    """Return a UTC ISO 8601 string for *now + remaining*.

    Negative durations are treated as zero (ETA is now).
    """
    if remaining.total_seconds() < 0:
        remaining = timedelta(0)
    eta = datetime.now(tz=timezone.utc) + remaining
    return eta.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(s: str) -> timedelta:
    """Parse a compact duration string into a :class:`timedelta`.

    Supported units: ``h`` (hours), ``m`` (minutes), ``s`` (seconds).
    At least one token is required.  Raises ``ValueError`` on invalid input.

    Examples::

        parse_duration("1h30m")    → timedelta(hours=1, minutes=30)
        parse_duration("90m")      → timedelta(minutes=90)
        parse_duration("0.5h")     → timedelta(minutes=30)
        parse_duration("1h30m45s") → timedelta(hours=1, minutes=30, seconds=45)
    """
    tokens = re.findall(r"([0-9]*\.?[0-9]+)\s*([hHmMsS])", s.strip())
    if not tokens:
        raise ValueError(
            f"Cannot parse duration {s!r}. "
            "Expected tokens like '1h', '30m', '45s', e.g. '1h30m' or '90m'."
        )
    hours = minutes = seconds = 0.0
    for value, unit in tokens:
        v = float(value)
        u = unit.lower()
        if u == "h":
            hours += v
        elif u == "m":
            minutes += v
        elif u == "s":
            seconds += v
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def duration_to_cron(s: str) -> str:
    """Return a one-shot 5-field cron expression (local time) for *now + duration*.

    The returned expression pins minute, hour, day-of-month, and month to the
    target local time, leaving day-of-week as ``*``.  Suitable for a one-shot
    ``CronCreate`` call (``recurring=False``).

    Seconds in the duration are rounded to the nearest minute (≥30s rounds up).

    Examples::

        duration_to_cron("1m")     → '41 13 24 3 *'  (1 minute from now, local)
        duration_to_cron("1h30m")  → '11 15 24 3 *'
        duration_to_cron("90m")    → '11 15 24 3 *'
    """
    remaining = parse_duration(s)
    # Round to nearest minute
    total_seconds = remaining.total_seconds()
    total_minutes = int(total_seconds // 60) + (1 if total_seconds % 60 >= 30 else 0)
    remaining_rounded = timedelta(minutes=total_minutes)

    target = datetime.now() + remaining_rounded  # local time
    return f"{target.minute} {target.hour} {target.day} {target.month} *"


def duration_to_iso8601(s: str) -> str:
    """Return a UTC ISO 8601 timestamp string for *now + duration*.

    Examples::

        duration_to_iso8601("1h30m") → '2026-03-24T12:15:00Z'
        duration_to_iso8601("90m")   → '2026-03-24T12:15:00Z'
    """
    return eta_iso(parse_duration(s))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eta.py",
        description="Compute an ETA from a duration string.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--iso8601",
        action="store_true",
        default=True,
        help="Output a UTC ISO 8601 timestamp (default).",
    )
    group.add_argument(
        "--cron",
        action="store_true",
        help="Output a one-shot 5-field cron expression in local time.",
    )
    parser.add_argument(
        "remaining",
        nargs="+",
        help="Duration string, e.g. '1h30m', '90m', '3600s'.",
    )
    args = parser.parse_args()

    raw = " ".join(args.remaining)
    try:
        if args.cron:
            print(duration_to_cron(raw))
        else:
            print(duration_to_iso8601(raw))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
