"""HTTP layer: one blueprint per resource group.

health (/api/health*), auth_api (/api/auth), pages (HTML pages), audio_api (/api/audio),
events_api, alerts_api, reviews_api, dashboard_api, reports_api (/api), live_api
(/api/live) and admin_api (/api/admin). Each module exposes ``bp``; src.app registers them
and /api/health reports any optional slice that failed to import.
"""

from __future__ import annotations

__all__ = ["pages", "health", "auth_api"]
