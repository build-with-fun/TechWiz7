# Jury interview preparation

Use these answers as a map to the actual repository. Show the recorded evidence when asked, and keep team-contribution answers tied to the team's own work log.

**What problem does this solve?** It gives a site operator a traceable first assessment of an uploaded or consented live sound, with both model opinions, audio quality, rules and a human review path. It does not dispatch emergency services.

**Why Flask, SQLite and plain JavaScript?** The app is a local/single-node prototype with server-rendered operations pages and modest event volume. Flask and SQLAlchemy keep the request-to-record path inspectable; SQLite avoids a separate database server for a live demo. Multiple workers and high-volume live capture would require a shared queue/state store and a server database.

**What is technically distinctive?** The two classifiers are separately trained and receive the same decoded audio independently. The comparison exposes disagreement and full class distributions instead of hiding them behind one label. Per-class rules and manual review connect model output to an operator action.

**How does the database work?** `audio_files` stores file provenance and a relative path; `events` stores the decision; `confidence_scores` keeps every model/class score; alerts, reviews, live windows, model versions and audit records link back to that event. See `DATABASE_SCHEMA.md` and `src/models.py`.

**How are data and API keys protected?** There is no paid classification API or frontend secret. Sessions use server-side role/capability checks and CSRF tokens; production mode requires a secret and secure cookies. Audio files live under a controlled storage root, outside static assets. Published demo accounts must be changed before public deployment.

**How are failures handled?** Invalid audio returns validation/quality errors, and absent model artifacts make analysis unavailable rather than inventing a second score. The UI shows unavailable and error states; alerts/reviews are written only for actual decisions. Tests cover the upload/live storage boundary.

**How is this different from a generic sound classifier?** It preserves both model distributions, comparison status, source evidence, configurable alert criteria, human decisions and an audit trail. The current measured model accuracy is below the SRS target, so the improvement is workflow transparency rather than a claim of superior detection.

**What trade-offs did you make?** A local SQLite/Flask architecture is easier to inspect and demo but does not prove 20,000-record concurrency or 99% uptime. The GTM transfer model was browser-trained on 32 distinct train parents per class; its measured server-path accuracy is 0.2778 on all 450 held-out clips. Browser/server feature parity remains unverified. These results rule out unsupervised safety use today.

**What would you improve?** First improve critical-class recall and test robustness on real site recordings, then complete GTM browser/server parity and performance measurements. Next, move live confirmation state into shared storage and establish an authorized dataset distribution.

**Which team member built which part?** Answer from the team's own commit history and work log. The repository's owner comments and generated documents are not evidence of individual contribution; do not invent names or days.

**How could it scale?** Keep the same API/data contracts, move SQLite to a server database, move live state and jobs to a shared queue, and load-test the actual throughput. No scaling benchmark has been recorded yet.
