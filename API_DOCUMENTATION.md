# HTTP API

The Flask app serves JSON under `/api` and HTML pages outside it. Most routes require a signed-in session. Browser state-changing requests must include the `X-CSRF-Token` value from the page's `csrf-token` meta tag; normal HTML forms include a hidden token. A capability check is enforced on the server for each protected action. Errors use an `error` object with `code`, `message`, optional `details`, and `request_id`; a missing model returns HTTP 503 rather than a fabricated analysis.

`GET/POST /register` creates a normal-user account through an HTML form; privileged roles remain administrator-assigned. `GET/POST /profile` lets a signed-in user update their own display name and email. Password changes use the JSON endpoint listed below.

## Core journeys

| Method and path | Purpose | Access |
|---|---|---|
| `POST /api/auth/login` | Sign in with `username` and `password`; creates a cookie session | Public |
| `POST /api/auth/logout` | End session | Signed in |
| `GET /api/auth/me` | Current user, role and capabilities | Signed in |
| `POST /api/auth/password` | Change own password using `current_password`, `new_password` | Signed in |
| `POST /api/audio/upload` | Multipart `file`; returns persisted analysis/event IDs | Upload capability |
| `GET /api/audio/<audio_pk>/download` | Download authorized stored audio | Owner or elevated audio access |
| `POST /api/live/sessions` | Start session; JSON must include `consent_ack: true` | Live capability |
| `POST /api/live/sessions/<id>/windows` | Submit a sequenced base64 WAV window | Session owner |
| `POST /api/live/sessions/<id>/stop` | Stop capture session | Session owner |
| `GET /api/live/sessions/<id>` | Session and window history | Session owner or elevated access |
| `GET /api/events` | Search visible events with filters | Signed in |
| `GET /api/events/<id>` | Event, model scores, decision and links | Owner or elevated event access |
| `GET /api/events/<id>/evidence` | Evidence metadata | Owner or elevated event access |
| `GET /api/dashboard/summary` | Role-scoped counts | Signed in |
| `GET /api/dashboard/timeline` | Chronological role-scoped activity | Signed in |

A typical upload uses `multipart/form-data`:

```text
POST /api/audio/upload
Content-Disposition: form-data; name="file"; filename="site-audio.wav"
Content-Type: audio/wav
```

A live window body contains `seq`, `audio_b64` (a complete WAV file), `sample_rate`, and `duration_sec`. The server checks the active session and persists both the window and linked event. The UI obtains microphone permission only after the user acknowledges consent.

## Operations and reporting

| Method and path | Purpose |
|---|---|
| `GET /api/alerts`, `/api/alerts/history`, `/api/alerts/<id>` | Alert queues and detail |
| `POST /api/alerts/<id>/acknowledge`, `/dismiss`, `/escalate` | Operator decision; dismissal requires a reason |
| `GET /api/reviews/queue`, `/api/reviews/history`, `/api/reviews/event/<id>` | Review worklists |
| `POST /api/reviews/<id>/decision` | Reviewer decision and comments |
| `GET /api/reports/event/<id>`, `/api/reports/period?from=YYYY-MM-DD&to=YYYY-MM-DD` | Event and date-range reports |
| `GET /api/export/events.csv`, `/api/export/events.xlsx` | Administrator-only filtered export |
| `GET /api/analytics/classes`, `/model-comparison`, `/consistency`, `/alerts`, `/quality`, `/reviews` | Stored-event aggregates |
| `GET /api/admin/config`, `PUT /api/admin/config/<file_key>` | View and atomically validate/edit JSON policy files |
| `GET /api/admin/users`, `POST /api/admin/users`, `PATCH /api/admin/users/<id>` | Administrator user management |
| `GET /api/admin/audit`, `/api/admin/monitoring/anomalies` | Audit and in-app health signals |
| `POST /api/admin/retention/preview`, `/api/admin/retention/purge?dry_run=0` | Preview or run a batch-limited retention purge |
| `GET /api/health`, `/api/health/ready`, `/api/health/detail`, `/api/version` | Liveness, database/model readiness, detailed health and build information; readiness returns 503 if the database or either model is unavailable |

Example validation response:

```json
{"error":{"code":"validation_error","message":"Request body must be a JSON object.","request_id":"..."}}
```

Route implementations in `src/api/` are the source of truth for all optional filters and payload fields. `documentation/api_contract.md` is an earlier proposal and includes routes that were never implemented; use this document for the delivered API. No model registration/activation API is currently available. `database/init_db.py` registers discovered artifacts, and persistence also records versions that produce real predictions. Registry activation does not hot-swap the saved model; restart after replacing an artifact.
