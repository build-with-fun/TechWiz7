"""Browser acceptance tests for the SRS items unit tests can't cover.

    .venv/bin/python tools/browser_acceptance.py            # own app on a temp database
    .venv/bin/python tools/browser_acceptance.py --base http://127.0.0.1:5055   # existing app

Without ``--base`` it starts its own app on a temporary database, so the demo data is
untouched. Chrome runs with a fake microphone that plays a WAV file, so the live page can
be tested without a person.

Checks (SRS section 1.6):
  ii        every seeded account signs in through the form; pages its role grants open,
            pages it lacks are refused
  iv        one upload per format (WAV, MP3, FLAC, OGG, M4A) through the Upload page
  v         a batch of good and bad files; each gets its own result or reason
  ix        player: play, pause, seek, volume by keyboard, replay
  li        an uncertain upload is marked "Manual Review Required"
  vi, lxiv  consent -> microphone Active -> windows classified by both models
  liv       the alert banner appears after consecutive critical windows
  vii       Available, Active, Paused, Disconnected and Permission denied states
  lxxix     nothing is captured before consent; Active is shown while listening

Writes ``reports/browser_acceptance.json`` and ``screenshots/acceptance/*.png``. Exits 1
if any check fails or a page throws an uncaught script error.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.auth import ROLE_GRANTS  # noqa: E402

CHROME = "/usr/bin/google-chrome"
SAMPLES = REPO / "sample_audio"
SHOTS = REPO / "screenshots" / "acceptance"
REPORT = REPO / "reports" / "browser_acceptance.json"

# Page -> required capability (src/api/pages.py). "/models/" only needs a login.
GUARDED_PAGES = {
    "/upload": "upload_audio",
    "/events": "view_own_events",
    "/dashboard": "view_own_events",
    "/live": "live_session",
    "/alerts": "view_alerts",
    "/reviews": "review_queue",
    "/analytics": "view_analytics",
    "/admin/config": "edit_config",
    "/admin/users": "manage_users",
}

# A different clip per format so none of the uploads count as duplicates.
FORMATS = [
    ("wav", "gunshot.wav", []),
    ("mp3", "glass_breaking.wav", ["-c:a", "libmp3lame", "-b:a", "128k"]),
    ("flac", "alarm_or_siren.wav", ["-c:a", "flac"]),
    ("ogg", "vehicle_horn.wav", ["-c:a", "libvorbis", "-q:a", "5"]),
    ("m4a", "animal_sound.wav", ["-c:a", "aac", "-b:a", "128k"]),
]

# Injected before the page scripts; they keep references to the microphone stream and
# the audio element so we can inspect them.
HOOKS = """
(() => {
  const md = navigator.mediaDevices;
  if (md && md.getUserMedia) {
    const original = md.getUserMedia.bind(md);
    md.getUserMedia = async (constraints) => {
      window.__acceptanceMicRequests = (window.__acceptanceMicRequests || 0) + 1;
      const stream = await original(constraints);
      window.__acceptanceStream = stream;
      return stream;
    };
  }
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    window.__acceptanceAudio = this;
    return play.apply(this, arguments);
  };
})();
"""


class Run:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.page_errors: list[str] = []
        self.console_errors: list[str] = []

    def check(self, fr: str, name: str, passed: bool, detail: str = "", shot: str = "") -> bool:
        row = {"fr": fr, "check": name, "passed": bool(passed), "detail": detail}
        if shot:
            row["screenshot"] = f"screenshots/acceptance/{shot}.png"
        self.checks.append(row)
        print(f"  {'PASS' if passed else 'FAIL'}  [{fr}] {name}" + (f": {detail}" if detail else ""),
              flush=True)
        return passed

    def watch(self, page, label: str) -> None:
        page.on("pageerror", lambda exc: self.page_errors.append(f"{label}: {exc}"))
        page.on("console", lambda msg: self.console_errors.append(f"{label}: {msg.text[:200]}")
                if msg.type == "error" else None)


# isolated app

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_isolated_app(workdir: Path) -> tuple[subprocess.Popen, str]:
    env = {**os.environ, "SST_DB_PATH": str(workdir / "acceptance.db"),
           "SST_STORAGE_DIR": str(workdir / "storage"), "SST_PRODUCTION": "0",
           "SST_LOAD_MODELS": "1"}
    subprocess.run([sys.executable, "database/init_db.py"], cwd=REPO, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    port = free_port()
    log = (workdir / "app.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"from src.app import create_app; create_app().run(host='127.0.0.1', port={port})"],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 240
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"app exited early; see {workdir / 'app.log'}")
        try:
            if requests.get(f"{base}/api/health/ready", timeout=2).status_code == 200:
                return proc, base
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("app did not become ready within 240 s")


def make_format_files(workdir: Path) -> list[tuple[str, Path]]:
    out = []
    for ext, source, codec in FORMATS:
        target = workdir / f"acceptance_{Path(source).stem}.{ext}"
        if ext == "wav":
            target.write_bytes((SAMPLES / source).read_bytes())
        else:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(SAMPLES / source), *codec, str(target)], check=True)
        out.append((ext, target))
    return out


# helpers

def accounts() -> list[dict]:
    doc = json.loads((REPO / "database" / "seed_credentials.json").read_text(encoding="utf-8"))
    return doc["users"]


def sign_in(page, base: str, username: str, password: str) -> bool:
    page.goto(f"{base}/login", wait_until="domcontentloaded")
    page.fill("#username", username)
    page.fill("#password", password)
    page.click("[data-login-submit]")
    page.wait_for_load_state("networkidle")
    return "/login" not in page.url


def wait_for_queue(page, count: int, timeout_s: float = 120) -> list[dict]:
    """Wait for ``count`` queue items to finish loading and return their results."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        items = page.evaluate("""() => Array.from(document.querySelectorAll('[data-upload-item]')).map(li => {
            const body = li.querySelector('.queue__body');
            return {name: li.getAttribute('data-upload-item'),
                    state: body ? (body.dataset.state || 'done') : 'done',
                    result: !!li.querySelector('.result'),
                    reason: (li.querySelector('.banner__text') || {}).innerText || '',
                    text: li.innerText.slice(0, 400)};
        })""")
        if len(items) >= count and all(i["state"] != "loading" for i in items):
            return items
        page.wait_for_timeout(500)
    raise TimeoutError(f"upload queue did not settle within {timeout_s:.0f} s")


def mic_state(page) -> str:
    return page.get_attribute("[data-mic-status]", "data-mic") or page.inner_text("[data-mic-status]")


def wait_for_mic(page, wanted: str, timeout_s: float = 20) -> str:
    deadline = time.time() + timeout_s
    state = ""
    while time.time() < deadline:
        state = mic_state(page)
        if state == wanted:
            return state
        page.wait_for_timeout(250)
    return state


# checks

def check_roles(run: Run, browser, base: str) -> None:
    for account in accounts():
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        run.watch(page, f"role:{account['username']}")
        ok = sign_in(page, base, account["username"], account["password"])
        run.check("i", f"{account['username']} signs in through the form", ok, page.url)
        grants = ROLE_GRANTS.get(account["role"], frozenset())
        wrong = []
        for path, capability in GUARDED_PAGES.items():
            response = page.goto(f"{base}{path}", wait_until="domcontentloaded")
            status = response.status if response else 0
            landed = page.url.split(base, 1)[-1].split("?")[0].rstrip("/") or "/"
            opened = status == 200 and landed == path.rstrip("/")
            if opened != (capability in grants):
                wrong.append(f"{path}: status {status}, landed {landed}, "
                             f"{'granted' if capability in grants else 'not granted'}")
        page.goto(f"{base}/dashboard", wait_until="networkidle")
        shot = f"role_{account['username']}_dashboard"
        page.screenshot(path=str(SHOTS / f"{shot}.png"), full_page=False)
        run.check("ii", f"{account['role']} ({account['username']}): pages match the permission matrix",
                  not wrong, "; ".join(wrong) or f"{len(GUARDED_PAGES)} pages checked", shot)
        context.close()


def check_uploads(run: Run, browser, base: str, admin: dict, files: list[tuple[str, Path]]) -> str:
    context = browser.new_context(viewport={"width": 1280, "height": 1000})
    page = context.new_page()
    run.watch(page, "upload")
    sign_in(page, base, admin["username"], admin["password"])
    event_id = ""
    for ext, path in files:
        page.goto(f"{base}/upload", wait_until="networkidle")
        page.set_input_files("#audio-input", str(path))
        items = wait_for_queue(page, 1)
        item = items[0]
        shot = f"upload_{ext}"
        page.screenshot(path=str(SHOTS / f"{shot}.png"), full_page=True)
        run.check("iv", f"{ext.upper()} upload analysed by both models",
                  item["result"] and "Python model" in item["text"] and "Teachable Machine" in item["text"],
                  item["text"].splitlines()[0] if item["text"] else "", shot)
        if not event_id:
            link = page.locator("[data-upload-item] a.btn").first
            href = link.get_attribute("href") if link.count() else ""
            event_id = (href or "").rstrip("/").rsplit("/", 1)[-1]

    batch = [SAMPLES / "machinery_fault.wav", SAMPLES / "invalid.notaudio",
             SAMPLES / "silence.wav", SAMPLES / "panic_scream.wav"]
    page.goto(f"{base}/upload", wait_until="networkidle")
    page.set_input_files("#audio-input", [str(p) for p in batch])
    items = {i["name"]: i for i in wait_for_queue(page, len(batch), 180)}
    page.screenshot(path=str(SHOTS / "upload_batch.png"), full_page=True)
    good = [items.get(p.name, {}).get("result") for p in (batch[0], batch[3])]
    bad = [items.get(p.name, {}) for p in (batch[1], batch[2])]
    bad_ok = all(b and not b["result"] and b["reason"].strip() for b in bad)
    run.check("v", "batch of 4: two analysed, two refused with a reason",
              all(good) and bad_ok,
              " | ".join(f"{n}: {'result' if i['result'] else i['reason'].splitlines()[0][:90]}"
                         for n, i in items.items()), "upload_batch")
    context.close()
    return event_id


def check_manual_review_wording(run: Run, browser, base: str, admin: dict) -> None:
    session = requests.Session()
    context = browser.new_context(viewport={"width": 1280, "height": 1000})
    page = context.new_page()
    run.watch(page, "review-wording")
    sign_in(page, base, admin["username"], admin["password"])
    for cookie in context.cookies():
        session.cookies.set(cookie["name"], cookie["value"])
    events = session.get(f"{base}/api/events?limit=50", timeout=30).json()
    rows = events.get("data") if isinstance(events, dict) else events
    if isinstance(rows, dict):
        rows = rows.get("events") or []
    flagged = [e for e in rows or [] if e.get("requires_manual_review")]
    if not flagged:
        run.check("li", "an uncertain event is marked Manual Review Required", False,
                  "no uploaded event needed review")
        context.close()
        return
    page.goto(f"{base}/events/{flagged[0]['id']}", wait_until="networkidle")
    page.screenshot(path=str(SHOTS / "manual_review_event.png"), full_page=False)
    run.check("li", "an uncertain event is marked Manual Review Required",
              page.get_by_text("Manual Review Required").count() > 0,
              f"event {flagged[0]['id']}: {flagged[0].get('consistency_status')}", "manual_review_event")
    context.close()


def check_player(run: Run, browser, base: str, admin: dict, event_id: str) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 1000})
    context.add_init_script(HOOKS)
    page = context.new_page()
    run.watch(page, "player")
    sign_in(page, base, admin["username"], admin["password"])
    page.goto(f"{base}/events/{event_id}", wait_until="networkidle")
    audio = "window.__acceptanceAudio"

    page.focus("[data-player-play]")
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)
    playing = page.evaluate(f"() => !!{audio} && !{audio}.paused && {audio}.currentTime > 0.3")
    run.check("ix", "play by keyboard", playing,
              f"time {page.inner_text('[data-player-time]')}")

    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    paused_at = page.evaluate(f"() => {audio}.paused ? {audio}.currentTime : -1")
    run.check("ix", "pause by keyboard", paused_at >= 0, f"paused at {paused_at:.2f} s")

    page.focus("[data-player-seek]")
    before = page.evaluate(f"() => {audio}.currentTime")
    for _ in range(5):
        page.keyboard.press("ArrowRight")
    page.keyboard.press("End")
    page.wait_for_timeout(400)
    after = page.evaluate(f"() => {audio}.currentTime")
    run.check("ix", "seek by keyboard", after > before + 0.5, f"{before:.2f} s -> {after:.2f} s")

    page.focus("[data-player-volume]")
    vol_before = page.evaluate(f"() => {audio}.volume")
    for _ in range(3):
        page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(200)
    vol_after = page.evaluate(f"() => {audio}.volume")
    run.check("ix", "volume by keyboard", vol_after < vol_before, f"{vol_before:.2f} -> {vol_after:.2f}")

    page.click("[data-player-replay]")
    page.wait_for_timeout(700)
    replay = page.evaluate(f"() => !{audio}.paused && {audio}.currentTime < 1.5")
    page.screenshot(path=str(SHOTS / "player.png"), full_page=False)
    run.check("ix", "replay restarts from 0:00 and plays", replay,
              f"time {page.inner_text('[data-player-time]')}", "player")
    context.close()


def check_live(run: Run, base: str, admin: dict, playwright) -> None:
    clip = SAMPLES / "person_asking_for_help.wav"
    browser = playwright.chromium.launch(
        executable_path=CHROME, headless=True,
        args=["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
              f"--use-file-for-fake-audio-capture={clip}", "--autoplay-policy=no-user-gesture-required"])
    context = browser.new_context(viewport={"width": 1280, "height": 1100}, permissions=["microphone"])
    context.add_init_script(HOOKS)
    page = context.new_page()
    run.watch(page, "live")
    sign_in(page, base, admin["username"], admin["password"])
    page.goto(f"{base}/live", wait_until="networkidle")

    available = wait_for_mic(page, "Available", 10)
    run.check("vii", "microphone shows Available before consent", available == "Available", available)
    requests_before = page.evaluate("() => window.__acceptanceMicRequests || 0")
    run.check("lxxix", "nothing is captured before consent", requests_before == 0,
              f"getUserMedia calls before consent: {requests_before}")
    run.check("lxxix", "the start button stays disabled until consent",
              page.is_disabled("[data-consent-grant]"))
    page.screenshot(path=str(SHOTS / "live_before_consent.png"), full_page=False)

    page.check("#consent")
    page.click("[data-consent-grant]")
    active = wait_for_mic(page, "Active", 20)
    run.check("vi", "consent opens a session and the microphone becomes Active", active == "Active",
              f"state {active}, title '{page.title()[:40]}'")

    deadline = time.time() + 60
    windows = 0
    while time.time() < deadline and windows < 3:
        windows = int(page.inner_text("[data-live-windows]") or 0)
        page.wait_for_timeout(500)
    python_class = page.inner_text("[data-model-card='python'] [data-model-class]")
    gtm_class = page.inner_text("[data-model-card='gtm'] [data-model-class]")
    page.screenshot(path=str(SHOTS / "live_active.png"), full_page=True)
    run.check("lxiv", "live windows are classified by both models", windows >= 3
              and python_class not in ("", "—") and gtm_class not in ("", "—"),
              f"{windows} windows; Python {python_class}, TM {gtm_class}", "live_active")

    alert_deadline = time.time() + 60
    while time.time() < alert_deadline and page.is_hidden("[data-live-alert]"):
        page.wait_for_timeout(500)
    alert_text = "" if page.is_hidden("[data-live-alert]") else page.inner_text("[data-live-alert]")
    page.screenshot(path=str(SHOTS / "live_alert.png"), full_page=True)
    run.check("liv", "a critical sound repeated across windows raises the live alert banner",
              bool(alert_text), alert_text[:160], "live_alert")

    page.click("[data-live-pause]")
    paused = wait_for_mic(page, "Paused", 5)
    page.screenshot(path=str(SHOTS / "live_paused.png"), full_page=False)
    run.check("vii", "Pause shows Paused", paused == "Paused", paused, "live_paused")
    page.click("[data-live-pause]")
    resumed = wait_for_mic(page, "Active", 5)
    run.check("vii", "Resume returns to Active", resumed == "Active", resumed)

    page.evaluate("() => window.__acceptanceStream.getAudioTracks()[0].dispatchEvent(new Event('ended'))")
    disconnected = wait_for_mic(page, "Disconnected", 5)
    page.screenshot(path=str(SHOTS / "live_disconnected.png"), full_page=False)
    run.check("vii", "a lost device shows Disconnected and stops monitoring",
              disconnected == "Disconnected" and page.is_disabled("[data-live-stop]"),
              disconnected, "live_disconnected")
    context.close()
    browser.close()

    denied_browser = playwright.chromium.launch(
        executable_path=CHROME, headless=True,
        args=["--no-sandbox", "--use-fake-device-for-media-stream", "--deny-permission-prompts"])
    context = denied_browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    run.watch(page, "live-denied")
    sign_in(page, base, admin["username"], admin["password"])
    page.goto(f"{base}/live", wait_until="networkidle")
    page.check("#consent")
    page.click("[data-consent-grant]")
    denied = wait_for_mic(page, "Permission denied", 15)
    page.screenshot(path=str(SHOTS / "live_permission_denied.png"), full_page=False)
    run.check("vii", "a refused permission shows Permission denied", denied == "Permission denied",
              denied, "live_permission_denied")
    context.close()
    denied_browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="", help="use a running app instead of an isolated one")
    args = parser.parse_args()

    SHOTS.mkdir(parents=True, exist_ok=True)
    for old in SHOTS.glob("*.png"):
        old.unlink()
    workdir = Path(tempfile.mkdtemp(prefix="sst-acceptance-"))
    proc = None
    base = args.base.rstrip("/")
    if not base:
        print("starting an isolated app on a scratch database ...", flush=True)
        proc, base = start_isolated_app(workdir)
    run = Run()
    admin = next(a for a in accounts() if a["role"] == "administrator")
    started = time.time()
    try:
        files = make_format_files(workdir)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=CHROME, headless=True,
                                                 args=["--no-sandbox",
                                                       "--autoplay-policy=no-user-gesture-required"])
            print("roles", flush=True)
            check_roles(run, browser, base)
            print("uploads", flush=True)
            event_id = check_uploads(run, browser, base, admin, files)
            print("manual review wording", flush=True)
            check_manual_review_wording(run, browser, base, admin)
            print("player", flush=True)
            if event_id:
                check_player(run, browser, base, admin, event_id)
            else:
                run.check("ix", "player", False, "no event to open")
            browser.close()
            print("live monitoring", flush=True)
            check_live(run, base, admin, playwright)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)

    run.check("-", "no uncaught script errors on any page", not run.page_errors,
              "; ".join(run.page_errors[:5]))
    failed = [c for c in run.checks if not c["passed"]]
    REPORT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base": "isolated scratch database" if proc is not None else base,
        "browser": "Google Chrome (headless), fake microphone playing "
                   "sample_audio/person_asking_for_help.wav",
        "duration_s": round(time.time() - started, 1),
        "passed": len(run.checks) - len(failed), "failed": len(failed),
        "checks": run.checks,
        "console_errors": run.console_errors,
        "page_errors": run.page_errors,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(run.checks) - len(failed)}/{len(run.checks)} checks passed -> "
          f"{REPORT.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
