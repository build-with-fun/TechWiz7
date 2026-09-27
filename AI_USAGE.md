# AI tool usage declaration: SonicSentinel AI

**Competition:** Aptech NextWave AI and ML, SonicSentinel AI
**SRS reference:** v1.0, deliverable 15, integrity rules §1.8
**Declaration date:** 2026-09-26

This file declares every use of artificial-intelligence tooling on the project, what it was
**not** used for, and our understanding of the competition's integrity rules. Each member
adds their own rows and signs at the bottom.

## 1. What was used

AI assistance was limited to **document summarization, some testing, research, and the
writing of code comments and documentation**. No AI tool produced a model, a dataset, a
prediction, or any part of the runtime decision logic.

We have probably taken research, code samples and parts of code from AI and from other
open-source projects as well, because we had a short time to build this.

## 2. Declaration of understanding

Every AI-assisted file above was read line by line, modified, and exercised before commit.

We understand that the **final decision in every case must not come from an external
generative AI API**, and that evaluators may demand an explanation of any function, inject a
defect for us to fix, or run hidden tests against this repository.

## 3. Signatures

| Member | Roll no. | Modules owned | Date | Signature |
|---|---|---|---|---|
| Ammar Ahmer | | Backend, model training and model making, documentation, deployment | 2026-09-26 | pending |
| Aimon | | Frontend, testing, other support | | |
| Khizr | | Suggestions, recordings, testing, implementation help | | |

## 4. Tools

| Tool | Used for |
|---|---|
| **xeno** (custom-built AI agents) | Research, docs, SRS summarization, testing, GitHub references |
| **Claude Code** | Code comments and testing |
| **ChatGPT, Cloudflare AI models** | Image generation, help and suggestions |
