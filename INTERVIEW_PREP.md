# Jury interview preparation

Short answers that point to the actual code. Show the recorded evidence when you can, and
answer questions about who did what from the team's own work log.

**What problem does this solve?** It gives an operator a first look at an uploaded or live
sound: what both models think, how good the audio is, which rule applies, and a way to get
a human to review it. It does not call emergency services.

**Why Flask, SQLite and plain JavaScript?** It is a single-machine prototype with
server-rendered pages and a modest number of events. Flask and SQLAlchemy keep the path from
request to database row easy to follow, and SQLite means no separate database server for
the demo. More workers or heavy live traffic would need a shared queue and a database server.

**What is technically interesting?** Two separately trained models get the same decoded
audio and never see each other's output. The comparison shows disagreements and both full
score lists instead of hiding them behind one label. Per-class rules and manual review turn
model output into an operator action.

**How does the database work?** `audio_files` stores where a file came from and its
relative path, `events` stores the decision, and `confidence_scores` keeps every score from
both models. Alerts, reviews, live windows, model versions and audit records all link back
to the event. See `DATABASE_SCHEMA.md` and `src/models.py`.

**How are data and keys protected?** There is no paid API and no secret in the frontend.
Permissions are checked on the server, forms use CSRF tokens, and production mode needs a
secret key and secure cookies. Audio is stored outside the static folder. The demo account
passwords must be changed before a public deployment.

**How are failures handled?** Bad audio gets a validation or quality error. If a model is
missing, analysis is unavailable instead of making up a second score. The UI has proper
empty and error states, and alerts and reviews are only written for real decisions.

**How is this different from a generic sound classifier?** It keeps both models' scores,
the comparison, the source audio, configurable alert rules, human decisions and an audit
trail. The Python model reaches 0.933 test accuracy and 0.933 macro-F1, with every
critical class at 85% recall or more, but the Teachable Machine model is well below target
(0.573 accuracy). So the strength is the transparent
workflow, not a claim of better detection.

**What trade-offs did you make?** SQLite and a single Flask process are simple to inspect
and demo. `tools/benchmark_scale.py` showed 20,000 events and 10 users at once staying under
1 s at the 95th percentile (`reports/scale.json`), but 99% uptime has not been measured over
a real evaluation period. The TM model was trained in the browser on 2,100 one-second
windows (one per training recording) for 200 epochs; above 1,400 samples TM needs Chrome
started with a larger JavaScript stack, or it overflows while preparing the data. Browser/server
spectrogram parity for TM is still unverified. None of this is ready for unsupervised
safety use.

**What would you improve?** Better recall on Aggression and Panic Scream using real site
recordings, confirm TM browser/server parity, move the live counter into shared storage,
and sort out a properly licensed way to share the dataset.

**Who built which part?** Answer from the commit history and your own work log. Generated
documents are not evidence of who did what; don't make up names or dates.

**How could it scale?** Keep the same API and data model, move SQLite to a database server,
put live state and jobs in a shared queue, and load-test again.
