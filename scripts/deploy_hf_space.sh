#!/usr/bin/env bash
# Publish the committed SonicSentinel AI tree to a Hugging Face Docker Space.
#
#   HF_TOKEN=hf_... scripts/deploy_hf_space.sh <owner>/<space-name>
#
# Before the first run: create the Space at https://huggingface.co/new-space (SDK: Docker,
# hardware: CPU basic, free), then under Settings -> Variables and secrets add a secret
# SST_SECRET_KEY (python -c "import secrets; print(secrets.token_hex(32))").
# Needs git and git-lfs. Only committed files are published; the audio dataset, notebooks,
# screenshots and documentation stay in the GitHub repository.
set -euo pipefail

SPACE="${1:?usage: HF_TOKEN=hf_... $0 <owner>/<space-name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"
git lfs version >/dev/null 2>&1 || { echo "git-lfs is required (sudo apt install git-lfs)"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$BUILD"
# Not needed to run the app: evidence, training archives and the SRS itself.
rm -rf "$BUILD"/audio_dataset "$BUILD"/notebooks "$BUILD"/screenshots "$BUILD"/diagrams \
       "$BUILD"/python_models/archive "$BUILD"/python_models/best_backup_baseline \
       "$BUILD"/python_models/best_transfer "$BUILD"/gtm_model/upload_package \
       "$BUILD"/gtm_model/teachable_machine_export.zip "$BUILD"/*.pdf
cp "$BUILD"/deploy/huggingface/Dockerfile "$BUILD"/Dockerfile
cp "$BUILD"/deploy/huggingface/SPACE.md "$BUILD"/README.md

cd "$BUILD"
git init -q -b main
git lfs install --local >/dev/null
# The Hub keeps binary files in LFS.
git lfs track "*.joblib" "*.h5" "*.bin" "*.pth" "*.npy" "*.zip" "*.png" "*.jpg" "*.woff2" \
              "*.ttf" "*.wav" "*.xlsx" >/dev/null
git add .gitattributes
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q \
    -m "SonicSentinel AI $(git -C "$ROOT" rev-parse --short HEAD)"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed $(git -C "$ROOT" rev-parse --short HEAD) to https://huggingface.co/spaces/${SPACE}"
echo "The Space builds for about 10-15 minutes; then open https://${SPACE/\//-}.hf.space/login"
