# SonicSentinel AI — HTTP API contract

**Version 1.0.0 · frozen 2026-09-23 · owner `sara` (backend)**
Consumers: `junaid` (pipeline + upload + live mic), `hina` (templates + fetch calls),
`imran` (deploy smoke test), `raheem` (acceptance traceability), `kamran` (security headers
+ load), `faris` (accessibility: error text must be readable aloud), `daniyal` (usability).

> **Status: proposed, not yet implemented.** Handlers are being written against this document.
> If something here does not fit what you are building, say so now — a change costs one edit
> in this file; after the handlers exist it costs a rewrite. Ask `sara` for a v1.1 section.

---

## 1. Ground rules

| Decision | Value | Why |
|---|---|---|
| Framework | **Flask 3.1** + Flask-Login + Flask-SQLAlchemy | Already installed in the shared `.venv`; FastAPI is a second stack nobody asked for. |
| Database | **SQLite** (file, WAL) | SRS permits it; single-node demo; zero-ops for the evaluator. |
| Auth | **Session cookie** (`Flask-Login`), `HttpOnly`, `SameSite=Lax`, `Secure` when served over TLS | Server-rendered pages means a cookie, not a bearer token. |
| Password storage | `werkzeug.security.generate_password_hash` (PBKDF2-SHA256) | Never plaintext, never reversible (FR i). |
| Pages | **Server-rendered Jinja** under `/` | Confirmed with `hina`: alert list and search are server-rendered with query params, not a JSON SPA. |
| API | JSON under `/api` | Used by the live-mic panel, the exports, and any fetch calls the templates need. |
| Media type | `application/json; charset=utf-8` | |
| Timestamps | ISO-8601 UTC with `Z`, e.g. `2026-09-23T20:06:19Z` | Store UTC, render local in the template. |
| IDs | integers for rows, UUID4 for live sessions | |
| Content type for writes | JSON, **except** uploads (multipart) | |

### 1.1 Error envelope — every failure, without exception

Successful responses return the resource directly (`200`) or `{"data": …, "meta": …}` for
lists. **Every** error — 4xx and 5xx — returns this shape:

```json
{
  "error": {
    "code": "invalid_credentials",
    "message": "That username and password do not match an account.",
    "details": {"field": "password"},
    "request_id": "b1c0f2e4a9d7"
  }
}
```

- `code` — stable, machine-readable, `snake_case`. Frontend may branch on it. **Never changes.**
- `message` — one plain sentence, safe to show a user and safe to read aloud (FR lxxvii,
  and `faris` needs it to make sense out of context). No stack traces, no SQL, no file paths,
  no library names, no internal identifiers. Exceptions are logged with `request_id`; the
  client is given only the `request_id` so support can correlate without leaking internals.
- `request_id` — also returned in the `X-Request-Id` response header on **every** response,
  success included.

**Codes in use:** `invalid_credentials` · `account_locked` · `not_authenticated` ·
`forbidden` · `not_found` · `validation_error` · `unsupported_media_type` · `file_too_large` ·
`duplicate_audio` · `quality_rejected` · `quality_unusable` · `invalid_state_transition` ·
`already_acknowledged` · `alert_not_acknowledgeable` · `config_invalid` · `export_too_large` ·
`model_unavailable` · `internal_error`

### 1.2 Roles and the permission matrix — FR ii

Five roles. `administrator` implies nothing automatically; the sets below are explicit and
each row is a test in `tests/test_api_auth_rbac.py`. **A wrong role gets `403`, not `404`
and not a hidden link** — hiding a link is not access control.

| Capability | normal user | audio reviewer | security operator | maintenance operator | administrator |
|---|:--:|:--:|:--:|:--:|:--:|
| Log in, view own profile | ✅ | ✅ | ✅ | ✅ | ✅ |
| Upload audio, start a mic session | ✅ | ✅ | ✅ | ✅ | ✅ |
| View events they created | ✅ | ✅ | ✅ | ✅ | ✅ |
| View **all** events, search/filter (FR lxvii) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Download/stream any event's audio | ❌ | ✅ | ✅ | ✅ | ✅ |
| View dashboards + analytics (FR lxviii) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Download event report / export CSV+XLSX (FR lxix, lxx) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Manual-review queue: view, decide, comment, override (FR lviii–lxi) | ❌ | ✅ | ❌ | ❌ | ✅ |
| View alerts, acknowledge/dismiss/escalate (FR liv–lvi) | ❌ | ❌ | ✅ | ❌ | ✅ |
| Alert history + false-alarm reporting | ❌ | ❌ | ✅ | ❌ | ✅ |
| Register a model version, set the active version (FR lxxv) | ❌ | ❌ | ❌ | ✅ | ✅ |
| Edit thresholds / alert rules / retention live (FR liii, lxxx) | ❌ | ❌ | ❌ | ✅ | ✅ |
| Manage users and roles (FR i, ii) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Read the audit trail (FR lxxvi) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Trigger a retention purge (FR lxxx) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Health / readiness | ✅ | ✅ | ✅ | ✅ | ✅ |

`maintenance operator` deliberately has **no** access to review decisions or alert
acknowledgement: they change how the system judges, not what it concluded. An evaluator
asking "can the maintenance operator close an alert?" must get `403`.

### 1.3 Status codes

`200` OK · `201` created · `202` accepted (async work queued) · `204` no content ·
`400` malformed body · `401` not authenticated · `403` authenticated but not permitted ·
`404` does not exist **or** you may not know it exists (see 1.4) · `409` conflict (duplicate,
already acknowledged, illegal state transition) · `413` file too large · `415` unsupported
type · `422` well-formed but rejected (unusable audio, config invalid) ·
`429` rate limited · `500` internal (never with a stack trace) · `503` model unavailable.

### 1.4 Two deliberate asymmetries

1. **`404` and `403` are both used, on purpose.** A request that must never reveal whether a
   row exists returns `404` (an unprivileged user asking for someone else's event). A request
   for an endpoint the role can simply never perform returns `403` (`faris` will test that the
   text reads sensibly aloud). The rule: *capability* → `403`; *existence of another user's
   data* → `404`.
2. **Config is validated on write and on read.** `PUT /api/admin/config/*` runs the same
   validator as boot, so an invalid rule set is `422` and the file on disk is untouched —
   the running system can never be left in a state that cannot start.

---

## 2. Endpoint index

`R` = required role. `—` = any authenticated user.

### Auth — FR i
| Method | Path | R | Purpose |
|---|---|---|---|
| POST | `/api/auth/login` | none | Authenticate, start a session |
| POST | `/api/auth/logout` | any | End the session |
| GET | `/api/auth/me` | any | Who am I, what may I do |
| POST | `/api/auth/password` | any | Change own password |

### Audio ingestion — FR iv–x, FR xxxvi, lxxi–lxxiv
| Method | Path | R | Purpose |
|---|---|---|---|
| POST | `/api/audio/upload` | any | Upload a clip; validate, hash, dedupe, classify |
| GET | `/api/events` | reviewer+ | Search/filter/paginate events (current user's own if normal user) |
| GET | `/api/events/<id>` | owner/reviewer+ | One event with both models' outputs |
| GET | `/api/events/<id>/audio` | owner/reviewer+ | Stream the stored clip |
| GET | `/api/events/<id>/evidence` | owner/reviewer+ | Why it was decided: features, confidences, quality, rule, config snapshot |
| POST | `/api/events/<id>/flag` | reviewer+ | Flag for investigation (blocks retention purge) |
| DELETE | `/api/events/<id>` | admin | Delete event + audio |
| POST | `/api/live/sessions` | any | Open a microphone session (FR lxxix consent gate) |
| POST | `/api/live/sessions/<sid>/windows` | owner | Push one rolling window; get live verdict |
| POST | `/api/live/sessions/<sid>/stop` | owner | Close the session, roll up results |
| GET | `/api/live/sessions/<sid>` | owner | Session state and events so far |

### Alerts — FR liii–lvi
| Method | Path | R | Purpose |
|---|---|---|---|
| GET | `/api/alerts` | security op+ | Alert list, filterable |
| GET | `/api/alerts/<id>` | security op+ | One alert with its evidence and rule |
| POST | `/api/alerts/<id>/acknowledge` | security op+ | Acknowledge (FR lv) |
| POST | `/api/alerts/<id>/dismiss` | security op+ | Dismiss as false alarm, with a reason |
| POST | `/api/alerts/<id>/escalate` | security op+ | Escalate, with a note |
| GET | `/api/alerts/history` | security op+ | All past alerts and their outcomes (FR lvi) |

### Manual review — FR lvii–lxi
| Method | Path | R | Purpose |
|---|---|---|---|
| GET | `/api/reviews/queue` | reviewer+ | Prioritised queue with the reason for each item |
| GET | `/api/reviews/<event_id>` | reviewer+ | Full context for a decision |
| POST | `/api/reviews/<event_id>/decision` | reviewer+ | Confirm/override; comments; preserves original |
| GET | `/api/reviews/history` | reviewer+ | Every decision, who and when |

### Dashboards, analytics, reports — FR lxiv–lxx
| Method | Path | R | Purpose |
|---|---|---|---|
| GET | `/api/dashboard/summary` | reviewer+ | Cards: totals, alerts open, queue depth, uptime |
| GET | `/api/dashboard/timeline` | reviewer+ | Events per bucket for the trend chart |
| GET | `/api/analytics/classes` | reviewer+ | Per-class counts and average confidence |
| GET | `/api/analytics/model-comparison` | reviewer+ | Agreement rate, disagreement, confidence-difference distribution |
| GET | `/api/analytics/consistency` | reviewer+ | Strong/Acceptable/Weak/Disagreement/Uncertain counts |
| GET | `/api/analytics/alerts` | reviewer+ | Alerts by severity, rate, false-alarm rate |
| GET | `/api/analytics/quality` | reviewer+ | Quality distribution and its effect on agreement |
| GET | `/api/analytics/reviews` | reviewer+ | Review outcomes, override rate, per-reviewer |
| GET | `/api/reports/event/<id>` | reviewer+ | Downloadable single-event report (HTML→PDF) |
| GET | `/api/reports/period` | reviewer+ | Downloadable period report (`from`, `to`) |
| GET | `/api/export/events.csv` | reviewer+ | CSV export honouring the active filters |
| GET | `/api/export/events.xlsx` | reviewer+ | Excel export, same filters |

### Models and administration — FR lxxv–lxxx
| Method | Path | R | Purpose |
|---|---|---|---|
| GET | `/api/models` | any | Both models, versions, active version, metrics |
| GET | `/api/models/<name>/versions` | any | Version history |
| POST | `/api/models/<name>/versions` | maint op+ | Register a trained version (FR lxxv) |
| POST | `/api/models/<name>/activate` | maint op+ | Switch the active version — never rewrites past results |
| GET | `/api/admin/config` | maint op+ | Every live config value, sources resolved |
| PUT | `/api/admin/config/thresholds` | maint op+ | Edit thresholds (validated) |
| PUT | `/api/admin/config/alert-rules` | maint op+ | Edit rules/severity/scale (validated) |
| PUT | `/api/admin/config/retention` | admin | Edit retention windows (FR lxxx) |
| GET | `/api/admin/config/history` | admin | Who changed which value, when, from what to what |
| GET | `/api/admin/users` | admin | List users |
| POST | `/api/admin/users` | admin | Create user with a role |
| PATCH | `/api/admin/users/<id>` | admin | Change role, enable/disable, reset password |
| GET | `/api/audit` | admin | Audit trail, filterable (FR lxxvi) |
| GET | `/api/monitoring/anomalies` | admin | FR lxxviii: error rate, latency, queue depth, disk |
| POST | `/api/admin/retention/preview` | admin | What a purge would delete (dry run) |
| POST | `/api/admin/retention/purge` | admin | Execute the purge; `?dry_run=1` by default |
| GET | `/api/health` | none | Liveness: DB reachable, which models are loaded |

---

## 3. Shapes

### 3.1 `Event` — the central resource

```json
{
  "id": 4821,
  "audio_id": "SST-2026-09-23-0004821",
  "filename": "warehouse_north_1402.wav",
  "source": "upload",
  "created_at": "2026-09-23T20:06:19Z",
  "created_by": {"id": 4, "username": "a.reviewer", "role": "audio_reviewer"},
  "duration_sec": 6.4,
  "sample_rate": 22050,
  "sha256": "9f2c…",
  "near_duplicate_of": null,
  "status": "Alert Generated",
  "quality": {"verdict": "Good", "score": 0.86, "detail": "low noise floor, no clipping"},
  "predicted_class": "Glass Breaking",
  "severity": "High",
  "severity_display": "High",
  "consistency_status": "Strong Match",
  "confidence_difference": 0.04,
  "requires_manual_review": false,
  "review_reason": null,
  "alert": {"id": 331, "status": "Open", "severity": "High"},
  "location": "Warehouse North — Bay 4",
  "models": {
    "python": {"name": "svm_mfcc_v3",    "version": "3.1.0", "predicted_class": "Glass Breaking", "confidence": 0.93},
    "gtm":    {"name": "teachable_machine_audio", "version": "2.0.0", "predicted_class": "Glass Breaking", "confidence": 0.89}
  }
}
```

- `status` is one of the SRS FR lxii set: `Uploaded`, `Classified`, `Uncertain`,
  `Alert Generated`, `Manual Review`, `Reviewed`, `Closed`.
- `quality.verdict` is one of FR xxxvii: `Good`, `Acceptable`, `Poor`, `Unusable`.
- `consistency_status` is one of FR xxxiii: `Strong Match`, `Acceptable Match`, `Weak Match`,
  `Model Disagreement`, `Uncertain Result`.
- `severity` is the stored value; `severity_display` is what to render after the
  `active_scale` mapping in `alert_rules/severity_levels.json` — **render the latter**.
- `confidence_difference` is `|python.confidence − gtm.confidence|` (FR xxxii).
- `models.*.version` is stored **per event** (FR lxxv). Activating a new version never
  changes these numbers on an existing row.
- `alert` is `null` when no alert was raised.

**List envelope**

```json
{"data": [ /* Event objects */ ],
 "meta": {"total": 20431, "page": 1, "per_page": 50, "pages": 409,
          "filters": {"severity": "High"}, "generated_at": "2026-09-23T20:06:19Z"}}
```

### 3.2 Search and filter — FR lxvii

`GET /api/events` accepts, all optional and all combinable:

| Param | Type | Notes |
|---|---|---|
| `q` | string | Free text over audio id, filename, location |
| `audio_id` | string | Exact |
| `filename` | string | Substring |
| `category` | string (repeatable) | One of the ten class names, **validated** against `config/classes.json` |
| `date_from`, `date_to` | ISO-8601 | Inclusive |
| `confidence_min`, `confidence_max` | 0..1 | Applied to the **top-class** confidence |
| `confidence_model` | `python` \| `gtm` \| `both` | Which model the confidence range applies to; default `both` |
| `severity` | string (repeatable) | Validated against the active scale |
| `quality` | string (repeatable) | Validated against FR xxxvii values |
| `review_status` | `none` \| `pending` \| `decided` | |
| `alert_status` | `open` \| `acknowledged` \| `dismissed` \| `escalated` \| `none` | |
| `consistency` | string (repeatable) | FR xxxiii values |
| `created_by` | username or id | Reviewer+ only |
| `source` | `upload` \| `microphone` | |
| `sort` | `created_at` \| `confidence` \| `severity` | default `created_at` |
| `order` | `asc` \| `desc` | default `desc` |
| `page`, `per_page` | int | `per_page` ≤ 200 |

An unknown value in a validated param returns `422 validation_error` naming the param and the
accepted values — silently ignoring a bad filter shows the user the wrong data and looks like
a bug in the search.

### 3.3 `POST /api/audio/upload` — the busiest endpoint

`multipart/form-data`: `file` (required), `location` (optional), `source` (optional, default
`upload`), `consent_ack` (`true` required when `source=microphone`, FR lxxix).

`201` response is the full `Event`. The pipeline behind it, in order:

1. size and type check → `413` / `415`
2. decode → `422 quality_unusable` if undecodable or under the configured minimum duration
3. **sha256 of the bytes** → exact duplicate → `409 duplicate_audio` naming the existing
   `audio_id`, unless `?allow_duplicate=true` (FR lxxiii)
4. **near-duplicate** check via perceptual fingerprint → creates the event and records
   `near_duplicate_of` (FR lxxiv). Never silently merged; a human decides.
5. preprocess + features (`taha`'s functions), quality verdict
6. Python model → `PredictionResult`; GTM model → `PredictionResult`
   (independent; the Python output is **never** an input to GTM — SRS 1.8)
7. confidence comparison, consistency status (`lorena`'s `classify_consistency`)
8. severity + recommended action from `alert_rules/`; repeated-detection confirmation
9. alert raised if the rule fires; manual review queued if a Step 17 condition matches
10. audit record; config snapshot stored with the event

`409` on a duplicate and `422` on unusable audio are **normal outcomes, not errors to hide** —
`hina` should render both inline without a red screen, and `daniyal` will test the wording.

### 3.4 `POST /api/live/sessions/<sid>/windows` — FR xxxvi

```json
// request
{"seq": 12, "captured_at": "2026-09-23T20:06:19Z", "duration_sec": 1.5, "audio_b64": "UklGR…"}
// 200 -- must answer inside the live budget (NFR: 1-3s window, <=3s)
{"seq": 12, "prediction": {"class": "Aggression", "confidence": 0.71},
 "consistency_status": "Acceptable Match", "quality": "Acceptable",
 "severity": "High", "severity_display": "High",
 "confirmed": false, "consecutive": 2, "needed": 3,
 "alert": null, "requeue_hint_ms": 500}
```

`confirmed` and `consecutive`/`needed` are exposed so the live panel can show
*"2 of 3 confirming windows"* rather than a spinner — the repeated-detection requirement
(FR xl) becomes visible instead of mysterious. `202` if the window arrived faster than the
configured minimum and was dropped; the body carries `requeue_hint_ms`.

### 3.5 `POST /api/reviews/<event_id>/decision` — FR lix–lxi

```json
// request
{"decision": "override",           // "confirm" | "override" | "reject"
 "final_class": "Machinery Fault",  // required when decision is "override"
 "final_severity": "Medium",        // optional; validated against the active scale
 "comments": "Bearing whine, no glass. Python model was misled by a broadband transient.",
 "false_alarm": false}
```

`200` echoes the event with **both** sets preserved:

```json
{"outcome": {"decision": "override", "final_class": "Machinery Fault",
             "final_severity": "Medium", "decided_by": "a.reviewer",
             "decided_at": "2026-09-23T20:10:02Z", "false_alarm": false},
 "original": {"python": {"predicted_class": "Glass Breaking", "confidence": 0.93},
              "gtm": {"predicted_class": "Glass Breaking", "confidence": 0.89}},
 "comments": "Bearing whine, no glass. …"}
```

`original` is immutable and stays readable forever (FR lxi): an override must never erase
what the models said, or the comparison report and the accuracy analytics become fiction.
`comments` is required for `override` and `reject` (`422` without it). A second decision on a
decided event is `409 invalid_state_transition`, not a silent overwrite.

### 3.6 `GET /api/evidence` inside an event — the "explain any function" defence

`GET /api/events/<id>/evidence` returns exactly what an evaluator needs to be shown on demand
(SRS 1.8):

```json
{"config_snapshot": {"thresholds_version": "1.0.0", "alert_rules_version": "1.0.0",
                     "severity_scale": "five_level", "content_hashes": {"thresholds": "31de2220aeb00aab", "…": "…"}},
 "rule_applied": {"class": "Glass Breaking", "severity": "High", "min_confidence": 0.60,
                  "min_top_two_margin": 0.10, "required_consecutive_detections": 3,
                  "requires_model_agreement": true, "matched_escalation": null},
 "features": {"version": "mfcc_v1", "n_features": 81, "preview": [0.12, -0.03, "…"]},
 "quality": {"verdict": "Good", "score": 0.86},
 "review_conditions_matched": [],
 "timeline": [{"at": "…", "actor": "system", "action": "classified", "note": "…"}]}
```

If an evaluator asks *"why did this become High?"* the answer is on this page, in the
configuration that was in force at the time.

---

## 4. Cross-cutting behaviour

- **Audit (FR lxxvi).** Logins (success and failure), uploads, microphone sessions and
  consent, predictions, alerts, acknowledgements, dismissals, escalations, review decisions
  and overrides, exports and downloads, model version registration and activation, config
  edits, retention purges, user and role changes. Each row: actor, action, target, timestamp,
  source IP, `request_id`, and before/after for any change.
- **Model versions (FR lxxv).** Every prediction stores the name **and** version of both
  models. `POST /api/models/<name>/activate` changes the default for *new* events only; a
  test proves an old event's stored versions and confidences are byte-identical afterwards.
- **Privacy (FR lxxix).** `POST /api/live/sessions` requires `consent_ack`; the consent event
  is audited; live-session audio is retained for the shorter configured window.
- **Retention (FR lxxx).** Configurable in `alert_rules/retention.json`; a purge is a dry run
  by default, never deletes an event on legal hold, and audits both the preview and the run.
- **Monitoring (FR lxxviii).** `/api/monitoring/anomalies` raises an in-app alert to
  administrators on error-rate, latency, queue-depth or disk thresholds.
- **Rate limits.** `429` with `Retry-After` on login (per IP and per username), upload and
  live windows. Failed logins lock the account for a configured window → `401` then `423`
  (`account_locked`).
- **Security headers** (owner `kamran`): `Content-Security-Policy` with no `unsafe-inline`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `X-Frame-Options: DENY`,
  `Strict-Transport-Security` when TLS.
- **NFR budgets** (owner `imran`): 30 s clip classified in ≤ 8 s; live window ≤ 3 s; the
  search endpoint stays under 200 ms with 20,000 events.

---

## 5. Who consumes what

| Consumer | Endpoints relied on |
|---|---|
| `junaid` | `POST /api/audio/upload`, the whole `/api/live/*` set, `GET /api/events*`, `/api/reports/period` |
| `hina` | all `GET`s, plus the review and alert `POST`s behind their buttons |
| `nadia`, `bilal`, `lorena` | `POST /api/models/<name>/versions`, `POST …/activate`, `GET /api/events/<id>/evidence` |
| `taha` | `quality` + `features` blocks in `Event` and evidence |
| `omar` | `POST /api/audio/upload` in bulk for the corpus; `GET /api/export/events.csv` |
| `imran` | `GET /api/health`, `POST /api/admin/retention/preview`, the NFR timings |
| `kamran` | security headers, `429`/`423` behaviour, search under load |
| `faris` | `message` text in every error; `403` vs `404` distinctions |
| `daniyal` | upload's `409`/`422` inline rendering; live-panel `consecutive`/`needed` |
| `raheem` | this table, as the acceptance traceability map |

---

## 6. Open questions — answers needed before handlers land

1. **`junaid`** — live mic: is the rolling-`POST` contract in §3.4 workable, or do you need a
   WebSocket? Rolling POST keeps state server-side and survives a page reload; a socket is
   smoother but adds a second transport to test and deploy. Default: rolling POST.
2. **`hina`** — confirm server-rendered Jinja with query params for search/filter (§3.2), so
   the filter state lives in the URL and is shareable and testable.
3. **`lorena`** — the exact callable and signature the app should use to obtain both
   `PredictionResult`s from a `PreprocessedAudio`, and where `save_bundle` output lands, so the
   app loads real models the same way the tests do.
4. **`taha`** — the function names for "validate a decoded clip" and "quality verdict", and
   whether preprocessing rejects unusable audio itself or returns a verdict for the caller.

Answer any of these and I move it into v1.1 the same hour. Silence means I build the
documented default.

---
*Change log — v1.0.0 (2026-09-23): first frozen contract. Framework, DB, auth, error envelope,
permission matrix, ~50 endpoints, shapes, cross-cutting behaviour, consumer map.*
