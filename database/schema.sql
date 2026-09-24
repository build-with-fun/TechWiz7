-- SonicSentinel AI -- canonical database schema.
--
-- GENERATED from src/models.py. Do not hand-edit this file: change the models and run
--   .venv/bin/python database/init_db.py --emit-schema
-- tests/test_database.py fails if this file and the ORM models disagree, so the two
-- definitions can never drift apart.
--
-- Dialect: SQLite 3. Datetimes are UTC, stored naive (no offset). See database/README.md.
-- Owner: sara.  SRS FR lxxi-lxxii, FR lxxv, FR lxxvi, FR lxxx.

CREATE TABLE users (
	id INTEGER NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	email VARCHAR(255), 
	display_name VARCHAR(128), 
	password_hash VARCHAR(255) NOT NULL, 
	role VARCHAR(32) NOT NULL, 
	is_active BOOLEAN NOT NULL, 
	failed_login_count INTEGER NOT NULL, 
	locked_until DATETIME, 
	must_change_password BOOLEAN NOT NULL, 
	password_changed_at DATETIME, 
	created_at DATETIME NOT NULL, 
	created_by_id INTEGER, 
	last_login_at DATETIME, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_users_role CHECK (role IN ('normal_user','audio_reviewer','security_operator','maintenance_operator','administrator')), 
	UNIQUE (email), 
	FOREIGN KEY(created_by_id) REFERENCES users (id)
);
CREATE INDEX ix_users_role ON users (role);
CREATE UNIQUE INDEX ix_users_username ON users (username);

CREATE TABLE audio_files (
	id INTEGER NOT NULL, 
	audio_id VARCHAR(32) NOT NULL, 
	filename VARCHAR(255) NOT NULL, 
	stored_path VARCHAR(512) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	perceptual_fingerprint VARCHAR(64), 
	near_duplicate_of_id INTEGER, 
	size_bytes INTEGER NOT NULL, 
	duration_sec FLOAT, 
	sample_rate INTEGER, 
	channels INTEGER, 
	container_format VARCHAR(16), 
	original_format VARCHAR(16), 
	source VARCHAR(16) NOT NULL, 
	location VARCHAR(255), 
	created_by_id INTEGER, 
	created_at DATETIME NOT NULL, 
	consent_acknowledged BOOLEAN NOT NULL, 
	consent_recorded_at DATETIME, 
	retention_expires_at DATETIME, 
	flagged_for_investigation BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_audio_source CHECK (source IN ('upload','microphone')), 
	FOREIGN KEY(near_duplicate_of_id) REFERENCES audio_files (id), 
	FOREIGN KEY(created_by_id) REFERENCES users (id)
);
CREATE UNIQUE INDEX ix_audio_files_audio_id ON audio_files (audio_id);
CREATE INDEX ix_audio_files_created_by_id ON audio_files (created_by_id);
CREATE INDEX ix_audio_files_dedupe ON audio_files (sha256, perceptual_fingerprint);
CREATE INDEX ix_audio_files_flagged_for_investigation ON audio_files (flagged_for_investigation);
CREATE INDEX ix_audio_files_location ON audio_files (location);
CREATE INDEX ix_audio_files_perceptual_fingerprint ON audio_files (perceptual_fingerprint);
CREATE INDEX ix_audio_files_retention_expires_at ON audio_files (retention_expires_at);
CREATE UNIQUE INDEX ix_audio_files_sha256 ON audio_files (sha256);
CREATE INDEX ix_audio_files_source ON audio_files (source);

CREATE TABLE audit_records (
	id INTEGER NOT NULL, 
	timestamp DATETIME NOT NULL, 
	actor_id INTEGER, 
	actor_username VARCHAR(64), 
	actor_role VARCHAR(32), 
	action VARCHAR(48) NOT NULL, 
	target_type VARCHAR(32), 
	target_id VARCHAR(64), 
	outcome VARCHAR(16) NOT NULL, 
	detail TEXT, 
	"before" JSON, 
	"after" JSON, 
	ip_address VARCHAR(45), 
	user_agent VARCHAR(255), 
	request_id VARCHAR(32), 
	sha256 VARCHAR(64), 
	PRIMARY KEY (id), 
	FOREIGN KEY(actor_id) REFERENCES users (id) ON DELETE SET NULL
);
CREATE INDEX ix_audit_records_action ON audit_records (action);
CREATE INDEX ix_audit_records_actor_id ON audit_records (actor_id);
CREATE INDEX ix_audit_records_actor_username ON audit_records (actor_username);
CREATE INDEX ix_audit_records_outcome ON audit_records (outcome);
CREATE INDEX ix_audit_records_request_id ON audit_records (request_id);
CREATE INDEX ix_audit_records_target_id ON audit_records (target_id);
CREATE INDEX ix_audit_records_target_type ON audit_records (target_type);
CREATE INDEX ix_audit_target ON audit_records (target_type, target_id);
CREATE INDEX ix_audit_when_action ON audit_records (timestamp, action);

CREATE TABLE live_sessions (
	id VARCHAR(36) NOT NULL, 
	user_id INTEGER NOT NULL, 
	device_label VARCHAR(128), 
	location VARCHAR(255), 
	status VARCHAR(16) NOT NULL, 
	started_at DATETIME NOT NULL, 
	ended_at DATETIME, 
	consent_acknowledged BOOLEAN NOT NULL, 
	consent_recorded_at DATETIME, 
	window_count INTEGER NOT NULL, 
	detection_count INTEGER NOT NULL, 
	alert_count INTEGER NOT NULL, 
	consecutive_state JSON, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_live_status CHECK (status IN ('active','stopped','expired')), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_live_sessions_status ON live_sessions (status);
CREATE INDEX ix_live_sessions_user_id ON live_sessions (user_id);

CREATE TABLE model_versions (
	id INTEGER NOT NULL, 
	model_name VARCHAR(16) NOT NULL, 
	version VARCHAR(32) NOT NULL, 
	label VARCHAR(128), 
	artifact_path VARCHAR(512), 
	feature_version VARCHAR(64), 
	algorithm VARCHAR(128), 
	metrics JSON, 
	trained_at DATETIME, 
	registered_at DATETIME NOT NULL, 
	registered_by_id INTEGER, 
	is_active BOOLEAN NOT NULL, 
	notes TEXT, 
	dataset_manifest_hash VARCHAR(64), 
	PRIMARY KEY (id), 
	CONSTRAINT uq_model_version UNIQUE (model_name, version), 
	CONSTRAINT ck_model_name CHECK (model_name IN ('python','gtm')), 
	FOREIGN KEY(registered_by_id) REFERENCES users (id)
);
CREATE INDEX ix_model_versions_is_active ON model_versions (is_active);
CREATE INDEX ix_model_versions_model_name ON model_versions (model_name);

CREATE TABLE events (
	id INTEGER NOT NULL, 
	audio_file_id INTEGER NOT NULL, 
	source VARCHAR(16) NOT NULL, 
	status VARCHAR(24) NOT NULL, 
	predicted_class VARCHAR(64), 
	final_class VARCHAR(64), 
	severity VARCHAR(24), 
	consistency_status VARCHAR(32), 
	confidence_difference FLOAT, 
	top_confidence FLOAT, 
	quality_verdict VARCHAR(16), 
	quality_score FLOAT, 
	quality_detail TEXT, 
	python_model_version_id INTEGER, 
	gtm_model_version_id INTEGER, 
	alert_rule_class VARCHAR(64), 
	requires_manual_review BOOLEAN NOT NULL, 
	review_reason TEXT, 
	config_snapshot JSON, 
	overlapping_classes JSON, 
	location VARCHAR(255), 
	live_session_id VARCHAR(36), 
	consecutive_detections INTEGER, 
	created_by_id INTEGER, 
	created_at DATETIME NOT NULL, 
	classified_at DATETIME, 
	updated_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_event_status CHECK (status IN ('Uploaded','Classified','Uncertain','Alert Generated','Manual Review','Reviewed','Closed')), 
	CONSTRAINT ck_event_quality CHECK (quality_verdict IS NULL OR quality_verdict IN ('Good','Acceptable','Poor','Unusable')), 
	CONSTRAINT ck_event_consistency CHECK (consistency_status IS NULL OR consistency_status IN ('Strong Match','Acceptable Match','Weak Match','Model Disagreement','Uncertain Result')), 
	FOREIGN KEY(audio_file_id) REFERENCES audio_files (id), 
	FOREIGN KEY(python_model_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(gtm_model_version_id) REFERENCES model_versions (id), 
	FOREIGN KEY(live_session_id) REFERENCES live_sessions (id), 
	FOREIGN KEY(created_by_id) REFERENCES users (id)
);
CREATE INDEX ix_events_audio_file_id ON events (audio_file_id);
CREATE INDEX ix_events_confidence_difference ON events (confidence_difference);
CREATE INDEX ix_events_consistency_status ON events (consistency_status);
CREATE INDEX ix_events_created_by_id ON events (created_by_id);
CREATE INDEX ix_events_final_class ON events (final_class);
CREATE INDEX ix_events_gtm_model_version_id ON events (gtm_model_version_id);
CREATE INDEX ix_events_live_session_id ON events (live_session_id);
CREATE INDEX ix_events_location ON events (location);
CREATE INDEX ix_events_predicted_class ON events (predicted_class);
CREATE INDEX ix_events_python_model_version_id ON events (python_model_version_id);
CREATE INDEX ix_events_quality_verdict ON events (quality_verdict);
CREATE INDEX ix_events_requires_manual_review ON events (requires_manual_review);
CREATE INDEX ix_events_review_queue ON events (requires_manual_review, status, created_at);
CREATE INDEX ix_events_search ON events (created_at, predicted_class, severity, status);
CREATE INDEX ix_events_severity ON events (severity);
CREATE INDEX ix_events_source ON events (source);
CREATE INDEX ix_events_status ON events (status);
CREATE INDEX ix_events_top_confidence ON events (top_confidence);

CREATE TABLE alerts (
	id INTEGER NOT NULL, 
	event_id INTEGER NOT NULL, 
	severity VARCHAR(24) NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	rule_class VARCHAR(64), 
	rule_snapshot JSON, 
	recommended_action TEXT, 
	message TEXT, 
	escalated_to_severity VARCHAR(24), 
	dedup_key VARCHAR(128), 
	created_at DATETIME NOT NULL, 
	acknowledged_by_id INTEGER, 
	acknowledged_at DATETIME, 
	resolved_by_id INTEGER, 
	resolved_at DATETIME, 
	resolution_note TEXT, 
	is_false_alarm BOOLEAN, 
	notified_channels JSON, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_alert_status CHECK (status IN ('Open','Acknowledged','Dismissed','Escalated','Closed')), 
	FOREIGN KEY(event_id) REFERENCES events (id), 
	FOREIGN KEY(acknowledged_by_id) REFERENCES users (id), 
	FOREIGN KEY(resolved_by_id) REFERENCES users (id)
);
CREATE INDEX ix_alerts_dedup_key ON alerts (dedup_key);
CREATE INDEX ix_alerts_event_id ON alerts (event_id);
CREATE INDEX ix_alerts_queue ON alerts (status, severity, created_at);
CREATE INDEX ix_alerts_severity ON alerts (severity);
CREATE INDEX ix_alerts_status ON alerts (status);

CREATE TABLE confidence_scores (
	id INTEGER NOT NULL, 
	event_id INTEGER NOT NULL, 
	model_name VARCHAR(16) NOT NULL, 
	model_version_id INTEGER, 
	class_name VARCHAR(64) NOT NULL, 
	confidence FLOAT NOT NULL, 
	rank INTEGER NOT NULL, 
	is_top BOOLEAN NOT NULL, 
	latency_sec FLOAT, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_confidence_cell UNIQUE (event_id, model_name, class_name), 
	CONSTRAINT ck_confidence_model CHECK (model_name IN ('python','gtm')), 
	CONSTRAINT ck_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0), 
	FOREIGN KEY(event_id) REFERENCES events (id), 
	FOREIGN KEY(model_version_id) REFERENCES model_versions (id)
);
CREATE INDEX ix_confidence_scores_class_name ON confidence_scores (class_name);
CREATE INDEX ix_confidence_scores_event_id ON confidence_scores (event_id);
CREATE INDEX ix_confidence_scores_is_top ON confidence_scores (is_top);
CREATE INDEX ix_confidence_scores_model_name ON confidence_scores (model_name);
CREATE INDEX ix_confidence_top ON confidence_scores (event_id, model_name, is_top);

CREATE TABLE live_windows (
	id INTEGER NOT NULL, 
	session_id VARCHAR(36) NOT NULL, 
	seq INTEGER NOT NULL, 
	captured_at DATETIME, 
	received_at DATETIME NOT NULL, 
	duration_sec FLOAT, 
	predicted_class VARCHAR(64), 
	python_class VARCHAR(64), 
	python_confidence FLOAT, 
	gtm_class VARCHAR(64), 
	gtm_confidence FLOAT, 
	confidence_difference FLOAT, 
	consistency_status VARCHAR(32), 
	quality_verdict VARCHAR(16), 
	severity VARCHAR(24), 
	confirmed BOOLEAN NOT NULL, 
	consecutive INTEGER, 
	needed INTEGER, 
	latency_ms FLOAT, 
	event_id INTEGER, 
	audio_file_id INTEGER, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_live_window_seq UNIQUE (session_id, seq), 
	FOREIGN KEY(session_id) REFERENCES live_sessions (id), 
	FOREIGN KEY(event_id) REFERENCES events (id), 
	FOREIGN KEY(audio_file_id) REFERENCES audio_files (id)
);
CREATE INDEX ix_live_windows_event_id ON live_windows (event_id);
CREATE INDEX ix_live_windows_session_id ON live_windows (session_id);

CREATE TABLE reviews (
	id INTEGER NOT NULL, 
	event_id INTEGER NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	priority VARCHAR(16) NOT NULL, 
	condition_ids JSON, 
	reason_text TEXT, 
	recommended_action TEXT, 
	queued_at DATETIME NOT NULL, 
	assigned_to_id INTEGER, 
	decision VARCHAR(16) NOT NULL, 
	final_class VARCHAR(64), 
	final_severity VARCHAR(24), 
	comments TEXT, 
	false_alarm BOOLEAN, 
	decided_by_id INTEGER, 
	decided_at DATETIME, 
	original_python_class VARCHAR(64), 
	original_python_confidence FLOAT, 
	original_gtm_class VARCHAR(64), 
	original_gtm_confidence FLOAT, 
	original_severity VARCHAR(24), 
	original_python_model_version VARCHAR(32), 
	original_gtm_model_version VARCHAR(32), 
	PRIMARY KEY (id), 
	CONSTRAINT ck_review_status CHECK (status IN ('Pending Review','In Review','Reviewed')), 
	CONSTRAINT ck_review_decision CHECK (decision IN ('confirm','override','reject','pending')), 
	FOREIGN KEY(event_id) REFERENCES events (id), 
	FOREIGN KEY(assigned_to_id) REFERENCES users (id), 
	FOREIGN KEY(decided_by_id) REFERENCES users (id)
);
CREATE INDEX ix_reviews_assigned_to_id ON reviews (assigned_to_id);
CREATE INDEX ix_reviews_decided_by_id ON reviews (decided_by_id);
CREATE INDEX ix_reviews_decision ON reviews (decision);
CREATE INDEX ix_reviews_event_id ON reviews (event_id);
CREATE INDEX ix_reviews_priority ON reviews (priority);
CREATE INDEX ix_reviews_queue ON reviews (status, priority, queued_at);
CREATE INDEX ix_reviews_status ON reviews (status);

-- Vocabulary enforced by CHECK constraints on this schema:
--   users.role              : normal_user, audio_reviewer, security_operator, maintenance_operator, administrator
--   events.status           : Uploaded, Classified, Uncertain, Alert Generated, Manual Review, Reviewed, Closed
--   events.quality_verdict  : Good, Acceptable, Poor, Unusable
--   events.consistency_status: Strong Match, Acceptable Match, Weak Match, Model Disagreement, Uncertain Result
--   alerts.status           : Open, Acknowledged, Dismissed, Escalated, Closed
--   model_versions.model_name: python, gtm
