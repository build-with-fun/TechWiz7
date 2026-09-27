#!/usr/bin/env python3
"""Check the UI in Chrome against a temporary database.

Needs Playwright and Chrome. Models are off; this checks navigation, the theme toggle,
layout at four widths and browser errors.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import WSGIRequestHandler, make_server  # noqa: E402

from database.init_db import seed_users  # noqa: E402
from src.app import create_app  # noqa: E402


class QuietHandler(WSGIRequestHandler):
    def log(self, *args, **kwargs):
        pass


def main() -> int:
    screenshots = REPO_ROOT / "screenshots" / "ui"
    screenshots.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Chromium; isolated seeded database; model inference disabled",
        "checks": [], "layouts": [], "browser_errors": [],
    }
    with TemporaryDirectory(prefix="sonicsentinel-ui-") as tmp:
        app = create_app(
            TESTING=True, SST_CSRF_ENABLED=True, SST_LOAD_MODELS=False,
            SST_DB_PATH=str(Path(tmp) / "ui.db"), SST_STORAGE_DIR=str(Path(tmp) / "storage"),
        )
        credentials = json.loads((REPO_ROOT / "database/seed_credentials.json").read_text())
        account = next(user for user in credentials["users"] if user["role"] == "administrator")
        with app.config["SST_SESSION_FACTORY"]() as session:
            seed_users(session, credentials, verbose=False)
            session.commit()
        server = make_server("127.0.0.1", 0, app, threaded=True, request_handler=QuietHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path="/usr/bin/google-chrome", headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                page = context.new_page()
                page.on("pageerror", lambda error: report["browser_errors"].append(str(error)))
                page.on("console", lambda message: report["browser_errors"].append(message.text)
                        if message.type == "error" else None)

                def visit(path):
                    response = page.goto(origin + path, wait_until="networkidle")
                    assert response.status == 200, (path, response.status)
                    page.wait_for_timeout(650)

                def capture(name):
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(screenshots / f"{name}.png"), full_page=True)

                def toggle_theme():
                    page.locator("[data-theme-toggle]").click()

                visit("/login")
                assert page.locator("html").get_attribute("data-theme") == "light"
                assert page.locator("[data-theme-toggle]").get_attribute("aria-pressed") == "true"
                capture("login-light")
                toggle_theme()
                assert page.locator("html").get_attribute("data-theme") == "dark"
                assert page.locator("[data-theme-toggle]").get_attribute("aria-pressed") == "false"
                page.reload(wait_until="networkidle")
                assert page.locator("html").get_attribute("data-theme") == "dark"
                assert 'data-theme="dark"' in context.request.get(origin + "/login").text()
                capture("login-dark")
                toggle_theme()
                page.keyboard.press("Tab")
                report["checks"].append("Dark/light toggle persists in the cookie and server-rendered HTML")

                page.locator("#username").fill(account["username"])
                page.locator("#password").fill(account["password"])
                page.locator("[data-login-submit]").click()
                page.wait_for_url("**/dashboard")
                report["checks"].append("Real sign-in form with CSRF protection")
                paths = page.locator('.nav a:not([target="_blank"])').evaluate_all(
                    "links => [...new Set(links.map(link => new URL(link.href).pathname))]"
                )
                for width in (320, 390, 768, 1440):
                    page.set_viewport_size({"width": width, "height": 900})
                    for path in paths:
                        visit(path)
                        layout = page.evaluate("""() => ({
                          width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
                          brokenImages: [...document.images].filter(i => !i.complete || !i.naturalWidth).length,
                          unnamedLinks: [...document.querySelectorAll('.nav a')].filter(a =>
                            !a.getAttribute('aria-label') && !a.textContent.trim()).length
                        })""")
                        report["layouts"].append({"path": path, **layout})
                        if path in ("/dashboard", "/upload", "/live") and width in (390, 1440):
                            capture(path.strip("/") + f"-{width}")
                    for path in ("/login?switch=1", "/register"):
                        visit(path)
                        layout = page.evaluate("() => ({width: innerWidth, scrollWidth: document.documentElement.scrollWidth})")
                        report["layouts"].append({"path": path, **layout})
                        if width == 390:
                            capture(path.split("?")[0].strip("/") + "-mobile")

                page.set_viewport_size({"width": 1440, "height": 1000})
                visit("/dashboard")
                toggle_theme()
                capture("dashboard-light")
                toggle_theme()
                report["checks"].append("All visible application navigation routes at 320, 390, 768, 1440 pixels")
                browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)
    report["layout_failures"] = [row for row in report["layouts"]
                                 if row["scrollWidth"] > row["width"] or row.get("brokenImages") or row.get("unnamedLinks")]
    report["passed"] = not report["browser_errors"] and not report["layout_failures"]
    destination = REPO_ROOT / "reports" / "ui_review.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "layouts"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
