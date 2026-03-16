"""eta.py — Compute an ETA timestamp from a remaining-time string.

CLI usage
---------
    python scripts/eta.py <remaining>

    <remaining> is a compact duration string composed of one or more
    ``<number><unit>`` tokens (case-insensitive, whitespace optional):

        1h30m        → 1 hour 30 minutes
        90m          → 90 minutes
        3600s        → 3600 seconds
        2h           → 2 hours
        1h 30m 45s   → 1 hour 30 minutes 45 seconds
        0.5h         → 30 minutes

    Prints a single UTC ISO 8601 timestamp, e.g. ``2026-03-16T18:45:00Z``.

Python usage
------------
    from eta import eta_iso
    from datetime import timedelta

    print(eta_iso(timedelta(hours=1, minutes=30)))
    # → '2026-03-16T18:45:00Z'
"""

from __future__ import annotations

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


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/eta.py <remaining>\n"
            "  <remaining>  duration string, e.g. '1h30m', '90m', '3600s'\n"
            "\n"
            "Prints the UTC ISO 8601 timestamp for now + remaining.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = " ".join(sys.argv[1:])
    try:
        remaining = parse_duration(raw)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(eta_iso(remaining))


if __name__ == "__main__":
    main()
