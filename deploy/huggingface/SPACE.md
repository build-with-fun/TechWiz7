---
title: SonicSentinel AI
emoji: 🔊
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# SonicSentinel AI — hosted demonstration

Sound-event detection with two independently trained models (a Python AST-embedding
classifier and a Google Teachable Machine audio model), a consistency check between them,
alert rules and a manual-review queue. Source, dataset instructions and evaluation:
see the project's GitHub repository.

Open the app at its direct address, `https://<owner>-<space>.hf.space/login`. Inside the
huggingface.co page frame the browser treats the login cookie as third-party and may drop it.

Demonstration accounts are listed in the repository README. The database lives inside the
container: it is recreated with the demo accounts whenever the Space restarts, and uploads
from an earlier run are not kept.
