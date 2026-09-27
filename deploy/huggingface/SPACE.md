---
title: SonicSentinel AI
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# SonicSentinel AI: hosted demo

Sound-event detection with two separately trained models (a Python classifier on AST
embeddings and a Google Teachable Machine audio model), a comparison between them, alert
rules and a manual-review queue. The code, dataset instructions and evaluation are in the
project's GitHub repository.

Open the app at its own address, `https://<owner>-<space>.hf.space/login`. Inside the
huggingface.co page frame the browser treats the login cookie as third-party and may drop it.

The demo accounts are listed in the repository README. The database lives inside the
container, so it is recreated with the demo accounts whenever the Space restarts, and
earlier uploads are lost.
