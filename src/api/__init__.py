"""The HTTP layer.

Owner: sara.  One blueprint per resource group, each in its own module, so a slice can be
added or replaced without touching the factory:

======================  ==========================  =========================
Module                  Blueprint url prefix        Serves
======================  ==========================  =========================
``health``              (none)                      ``/api/health*``, ``/api/version``
``auth_api``            ``/api/auth``               sign-in, sign-out, ``/me``, password
``pages``               (none)                      ``/`` page routes and ``/api/events/<id>/visuals``
``audio_api``           ``/api/audio``              upload, dedupe, stored audio
``events_api``          ``/api``                    event detail, evidence, visuals, search
``alerts_api``          ``/api``                    alerts, acknowledgement, history
``reviews_api``         ``/api``                    manual-review queue and decisions
``dashboard_api``       ``/api``                    aggregates, analytics
``reports_api``         ``/api``                    report and CSV/Excel export
``live_api``            ``/api/live``               microphone sessions
``admin_api``           ``/api/admin``              configuration, users, audit, models
======================  ==========================  =========================

The slices after ``pages`` are optional and are discovered by ``src.app._register_blueprints``
-- a module that is not there yet is logged and skipped rather than breaking the boot. That is
deliberate while several engineers are landing endpoints in parallel.

Every registered blueprint must expose ``bp``. Routes that render a page live in ``pages``;
routes that return data live in the ``*_api`` modules. A route that returns HTML *and* JSON
depending on ``Accept`` is a route that has two untested halves -- it is not done here.
"""

from __future__ import annotations

__all__ = ["pages", "health", "auth_api"]
