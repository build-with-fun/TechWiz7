"""Take screenshots of the current UI.

Usage:
    .venv/bin/python tools/capture_screenshots.py [--out screenshots] [--base http://127.0.0.1:5055]

Same viewports, logins and data every time. Fails if a page shows an error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent

CREDS = [
    ("admin", "Admin#Sonic2026", "administrator"),
    ("evaluator", "Eval#Sonic2026", "administrator"),
    ("reviewer", "Review#Sonic2026", "audio_reviewer"),
    ("operator", "Operate#Sonic2026", "security_operator"),
    ("user", "User#Sonic2026", "normal_user"),
]

# Desktop plus the two CSS breakpoints.
VIEWPORTS = [
    ("desktop", 1440, 900),
    ("tablet", 900, 1100),
    ("mobile", 390, 844),
]

# (filename, path, username, extra wait for charts). Note the trailing slash on /models/,
# otherwise it redirects. Public pages are captured signed out.
PUBLIC_PAGES = [
    ("00_home", "/", 2600, True),
    ("01_login", "/login", 800, False),
    ("14_register", "/register", 800, False),
]

PAGES = [
    ("02_dashboard", "/dashboard", "admin", 1500),
    ("03_upload", "/upload", "admin", 800),
    ("04_events", "/events", "admin", 1200),
    ("05_event_detail", None, "admin", 1200),  # resolved from the events list
    ("06_live", "/live", "admin", 1000),
    ("07_alerts", "/alerts", "admin", 1000),
    ("08_reviews", "/reviews", "reviewer", 1000),
    ("09_analytics", "/analytics", "admin", 1800),
    ("10_models", "/models/", "admin", 1000),
    ("11_admin_config", "/admin/config", "admin", 800),
    ("12_admin_users", "/admin/users", "admin", 800),
    ("13_profile", "/profile", "admin", 500),
    # Same pages seen by different roles.
    ("15_evaluator_dashboard", "/dashboard", "evaluator", 1500),
    ("16_operator_alerts", "/alerts", "operator", 1000),
    ("17_user_dashboard", "/dashboard", "user", 1500),
]


def csrf_token(session: requests.Session, base: str) -> str:
    page = session.get(f"{base}/login", timeout=30)
    page.raise_for_status()
    import re

    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    if not match:
        raise RuntimeError("login page has no csrf_token field")
    return match.group(1)


def login(base: str, session: requests.Session, username: str, password: str) -> None:
    token = csrf_token(session, base)
    response = session.post(
        f"{base}/session",
        data={"username": username, "password": password, "csrf_token": token},
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()
    if "Sign in" in response.text and "csrf_token" in response.text:
        raise RuntimeError(f"login as {username} failed")


def first_event_id(base: str, session: requests.Session) -> str:
    """Pick an event to open (as admin, who can see all events)."""
    response = session.get(f"{base}/api/events?limit=1", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload.get("data"), list):
        events = payload["data"]
    elif isinstance(payload.get("data"), dict):
        events = payload["data"].get("events") or []
    else:
        events = payload.get("events") or []
    if not events:
        return ""
    return str(events[0]["id"])


def admin_event_id(base: str) -> str:
    """Event id for the event detail capture, looked up once as admin."""
    session = requests.Session()
    login(base, session, "admin", "Admin#Sonic2026")
    return first_event_id(base, session)


def fail_on_browser_error(page) -> None:
    """Fail if the page shows an error."""
    problems = []
    for problem in page.query_selector_all("text=/Exception|Traceback|Internal Server Error/"):
        problems.append(problem.inner_text()[:200])
    if problems:
        raise RuntimeError(f"page rendered an error: {problems[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="screenshots", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:5055")
    parser.add_argument("--roles", action="store_true", help="also capture per-role variants")
    args = parser.parse_args()

    out: Path = (REPO / args.out).resolve()
    # Only clear screenshots/ui; the other folders hold TM and acceptance screenshots.
    ui = out / "ui"
    if ui.exists():
        shutil.rmtree(ui)
    ui.mkdir(parents=True)

    response = requests.get(f"{args.base}/api/health/ready", timeout=30)
    if response.status_code != 200:
        print(f"server not ready at {args.base}: {response.status_code}", file=sys.stderr)
        return 1
    if not json.loads(response.text).get("analysis_ready"):
        print("server reports analysis not ready", file=sys.stderr)
        return 1

    sessions = {}
    for username, password, _role in CREDS:
        session = requests.Session()
        login(args.base, session, username, password)
        sessions[username] = session

    event_id = admin_event_id(args.base)
    if not event_id:
        print("no events visible to admin; skipping 05_event_detail", file=sys.stderr)

    captured = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])

        # One browser context per role and viewport.
        for vp_name, width, height in VIEWPORTS:
            for username, _password, _role in CREDS:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    base_url=args.base,
                )
                session = sessions[username]
                # Playwright wants url or domain+path, not both.
                context.add_cookies(
                    [
                        {"name": name, "value": value, "url": args.base}
                        for name, value in session.cookies.get_dict().items()
                    ]
                )

                page = context.new_page()
                page.set_default_timeout(20000)

                # Admin on desktop gets the dark theme, tablet the light one.
                if vp_name == "desktop" and username == "admin":
                    page.add_init_script(
                        "localStorage.setItem('sst-theme', 'dark');"
                    )
                elif vp_name == "tablet":
                    page.add_init_script(
                        "localStorage.setItem('sst-theme', 'light');"
                    )

                for filename, path, role, wait_ms in PAGES:
                    if role is not None and role != username:
                        continue
                    target = path if path else f"/events/{event_id}"
                    try:
                        page.goto(target, wait_until="networkidle")
                    except Exception as exc:
                        print(f"  ! {filename}: navigation failed: {exc}", file=sys.stderr)
                        continue
                    if page.title().startswith("Error"):
                        print(f"  ! {filename}: error page title", file=sys.stderr)
                        continue
                    fail_on_browser_error(page)
                    overflow = page.evaluate(
                        "() => document.documentElement.scrollWidth"
                        " - document.documentElement.clientWidth"
                    )
                    if overflow > 0:
                        print(
                            f"  ! {filename}: horizontal overflow of {overflow}px",
                            file=sys.stderr,
                        )
                    time.sleep(wait_ms / 1000)

                    suffix = "" if vp_name == "desktop" else f"_{vp_name}"
                    stem = f"{filename}{suffix}"
                    page.screenshot(path=str(ui / f"{stem}.png"), full_page=False)
                    print(f"  + {stem}.png ({username}, {width}x{height})")
                    captured += 1

                context.close()

            # Signed out: the product page, sign-in and sign-up.
            context = browser.new_context(viewport={"width": width, "height": height}, base_url=args.base)
            page = context.new_page()
            page.set_default_timeout(20000)
            for filename, path, wait_ms, full in PUBLIC_PAGES:
                page.goto(path, wait_until="networkidle")
                if full:
                    # Scroll through so the scroll animations run.
                    total = page.evaluate("() => document.documentElement.scrollHeight")
                    for y in range(0, total, 500):
                        page.evaluate(f"() => window.scrollTo(0, {y})")
                        page.wait_for_timeout(120)
                    page.evaluate("() => window.scrollTo(0, 0)")
                page.wait_for_timeout(wait_ms)
                fail_on_browser_error(page)
                suffix = "" if vp_name == "desktop" else f"_{vp_name}"
                page.screenshot(path=str(ui / f"{filename}{suffix}.png"), full_page=full)
                print(f"  + {filename}{suffix}.png (signed out, {width}x{height})")
                captured += 1
            context.close()

    print(f"\ncaptured {captured} screenshots into {out.relative_to(REPO)}/ui/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
