"""Train and export the Teachable Machine audio model through its web UI.

Needs Playwright and Chrome. The ZIPs come from audio_dataset/scripts/make_gtm_imports.py
(training recordings only).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "gtm_model/upload_package/tm_imports/index.json"
EXPORT = ROOT / "gtm_model"
# Screenshots for SRS deliverable 5.
SHOTS = ROOT / "screenshots" / "gtm"
# On Intel + NVIDIA laptops Chrome uses the Intel GPU, where 2,100 samples never finished
# preparing. These PRIME variables move WebGL to the NVIDIA card.
NVIDIA_OFFLOAD = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(EXPORT),
                        help="folder for metadata.json, model.json and weights.bin "
                             "(use gtm_model/candidates/<name> to compare before serving)")
    parser.add_argument("--nvidia-offload", action="store_true",
                        help="run Chrome's WebGL on the NVIDIA GPU of a hybrid-graphics laptop")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Advanced > Epochs in TM (0 keeps TM's default of 50)")
    parser.add_argument("--tag", default="",
                        help="added to screenshot names so several runs on one day are all kept")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d") + (f"_{args.tag}" if args.tag else "")
    classes = json.loads(INDEX.read_text(encoding="utf-8"))
    ordered = [next(item for item in classes if item["class"] == "Background Noise")]
    ordered += [item for item in classes if item["class"] != "Background Noise"]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/bin/google-chrome", headless=True,
            env={**os.environ, **NVIDIA_OFFLOAD} if args.nvidia_offload else None,
            # Headless Chrome uses software WebGL by default, which was far too slow.
            # ANGLE over desktop GL uses the GPU.
            args=["--no-sandbox", "--disable-dev-shm-usage", "--enable-gpu",
                  "--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page.on("console", lambda message: print("browser error:", message.text[:250], flush=True)
                if message.type == "error" else None)
        page.goto("https://teachablemachine.withgoogle.com/train/audio", wait_until="domcontentloaded")
        page.wait_for_timeout(5500)
        page.mouse.click(416, 464)  # close the intro tour

        for index, item in enumerate(ordered):
            if index > 1:
                page.get_by_text("Add a class", exact=True).click()
            if index > 0:
                page.locator('svg[aria-label="Edit class name"]').nth(index).click()
                field = page.locator("input#label-input").nth(index)
                field.fill(item["class"])
                field.press("Enter")
            page.locator('input[type=file][accept="application/zip"]').nth(index).set_input_files(
                str(ROOT / item["zip"])
            )
            # Loading takes longer with more samples, so scale the wait.
            loaded_from = time.time()
            page.get_by_text("Loading sample...", exact=False).wait_for(
                state="hidden", timeout=max(120000, item["sample_count"] * 3000)
            )
            print("Imported", item["class"], item["sample_count"],
                  f"in {time.time() - loaded_from:.0f}s", flush=True)

        if args.epochs:
            # The second tour card covers the Train button. Hide it (removing it breaks TM).
            page.locator("tm-onboard-box").evaluate_all(
                "els => els.forEach(e => { e.style.visibility = 'hidden'; e.style.pointerEvents = 'none'; })")
            page.get_by_text("Advanced", exact=True).first.click()
            page.wait_for_timeout(800)
            epochs = page.locator("input[type=number]:visible").first
            epochs.fill(str(args.epochs))
            epochs.press("Tab")
            print("Epochs set to", epochs.input_value(), flush=True)
        page.screenshot(path=str(SHOTS / f"{stamp}_01_classes_imported.png"), full_page=True)
        print("buttons before training", [x.strip() for x in page.locator("button").all_text_contents()][-25:], flush=True)
        # Clicks can be ignored while TM is still loading, so check and retry.
        for attempt in range(1, 5):
            page.get_by_role("button", name="Train Model").last.click()
            page.wait_for_timeout(3000)
            labels = [x.strip() for x in page.locator("button").all_text_contents()]
            if any("Training" in x for x in labels) or "Model Trained" in labels:
                break
            print(f"Train click {attempt} did not start training; retrying", flush=True)
        else:
            page.screenshot(path=str(SHOTS / f"{stamp}_train_not_started.png"), full_page=True)
            raise RuntimeError("Teachable Machine did not start training")
        print("Training started", flush=True)
        # Up to an hour (1,400 samples took about 5 minutes on the GPU).
        started = time.time()
        for interval in range(240):
            page.wait_for_timeout(15000)
            labels = [label.strip() for label in page.locator("button").all_text_contents()]
            # Log TM's progress text to tell slow from stuck.
            import re as _re

            # TM uses shadow DOM; text locators can see into it.
            status = page.get_by_text(_re.compile(r"Preparing training data|Epoch")).all_inner_texts()[-2:]
            if interval % 20 == 19:
                page.screenshot(path=str(SHOTS / f"{stamp}_progress_{interval + 1:03d}.png"))
            print("Training interval", interval + 1, f"{time.time() - started:.0f}s", status,
                  [label for label in labels if "Train" in label or "Export" in label], flush=True)
            if "Model Trained" in labels:
                break
        else:
            page.screenshot(path=str(SHOTS / f"{stamp}_timeout.png"), full_page=True)
            raise TimeoutError("Teachable Machine did not finish training within 60 minutes")
        print(f"Training finished after {time.time() - started:.0f}s", flush=True)

        page.screenshot(path=str(SHOTS / f"{stamp}_02_trained.png"), full_page=True)
        page.get_by_role("button", name="Export Model").click()
        page.wait_for_timeout(2500)
        radios = page.get_by_role("radio", name="Download")
        if radios.count():
            radios.last.check(force=True)
        else:
            page.get_by_text("Download", exact=True).last.click()
        page.wait_for_timeout(1000)
        print("export buttons", [x.strip() for x in page.locator("button").all_text_contents()][-30:], flush=True)
        print("export radios", page.locator('input[type="radio"]').evaluate_all(
            "els => els.map(e => e.outerHTML.slice(0,300))"), flush=True)
        page.screenshot(path=str(SHOTS / f"{stamp}_03_export_dialog.png"), full_page=True)
        try:
            with page.expect_download(timeout=120000) as downloaded:
                page.get_by_role("button", name="Download my model").click()
        except Exception as exc:
            page.screenshot(path=str(SHOTS / f"{stamp}_export_failed.png"), full_page=True)
            browser.close()
            raise RuntimeError(f"Export did not download; see {SHOTS}. ({exc})") from exc
        archive_path = out / "teachable_machine_export.zip"
        downloaded.value.save_as(archive_path)
        print("Exported", archive_path, flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            print("Contents", archive.namelist(), flush=True)
            for entry in archive.infolist():
                name = Path(entry.filename).name
                if name in {"metadata.json", "model.json", "weights.bin"}:
                    (out / name).write_bytes(archive.read(entry))
        page.screenshot(path=str(SHOTS / f"{stamp}_04_exported.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
