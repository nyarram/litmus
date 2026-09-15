"""Score-over-time dashboard for litmus eval reports.

The filesystem is the database: the dashboard reads ``*.json`` report files
from a directory (``reports/`` by convention) and renders runs, per-case
detail, and scorer trends. No database to operate; point it at a directory
and it works.
"""

from litmus.dashboard.app import create_app

__all__ = ["create_app"]
