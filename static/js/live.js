/**
 * live.js — the live microphone monitor (FR xxxvi, FR lxxix).
 *
 * The flow, and why each step is where it is:
 *   1. Consent gate. A microphone session may only start behind an explicit
 *      acknowledgement (FR lxxix) — the server refuses POST /api/live/sessions
 *      without consent_ack, so the UI disables Start until the checkbox is on.
 *   2. Capture. getUserMedia opens the mic; AudioWorklet-free ScriptProcessor
 *      keeps the code small. We record exactly one window of live_window_sec
 *      seconds (config-driven), encode it as 16-bit PCM WAV in the browser,
 *      and POST it base64 to /api/live/sessions/<id>/windows.
 *   3. Render. One window in, one verdict out: both models' class/confidence,
 *      the |python − gtm| difference, the consistency verdict, the confirmed
 *      streak — the same record an upload produces, so the two screens agree.
 *   4. Teardown. The stream, the timer and the audio context are registered
 *      with SST.lifecycle, so pagehide/overnight tabs cannot leak a mic.
 *
 * DOM contract (from live.html):
 *   [data-consent-panel] [data-consent-check] [data-consent-grant]
 *   [data-live-root] [data-live-state] [data-live-session-id]
 *   [data-live-start] [data-live-stop] [data-live-mute]
 *   [data-live-meter-fill] [data-live-meter-peak]
 *   [data-live-class] [data-live-consistency] [data-live-difference]
 *   [data-model-card=python|gtm] > [data-model-class] [data-model-confidence] [data-model-bar] [data-model-note]
 *   [data-live-windows] [data-live-events] [data-scale-fill] [data-scale-count] (per class li)
 *   [data-live-tape]
 */
"use strict";

(function (SST) {
  var root = document.querySelector("[data-live-root]");
  if (!root || typeof SST === "undefined") {
    return;
  }

  var byId = SST.byId || function (id) { return document.getElementById(id); };
  var el = SST.el;
  var analysisReady = root.getAttribute("data-analysis-ready") === "1";

  var consentPanel = document.querySelector("[data-consent-panel]");
  var consentCheck = document.querySelector("[data-consent-check]");
  var consentGrant = document.querySelector("[data-consent-grant]");
  var stateLabel = document.querySelector("[data-live-state]");
  var sessionLabel = document.querySelector("[data-live-session-id]");
  var startBtn = document.querySelector("[data-live-start]");
  var stopBtn = document.querySelector("[data-live-stop]");
  var muteBtn = document.querySelector("[data-live-mute]");
  var meterFill = document.querySelector("[data-live-meter-fill]");
  var meterPeak = document.querySelector("[data-live-meter-peak]");
  var verdictClass = document.querySelector("[data-live-class]");
  var verdictConsistency = document.querySelector("[data-live-consistency]");
  var differenceValue = document.querySelector("[data-live-difference]");
  var windowsCount = document.querySelector("[data-live-windows]");
  var eventsCount = document.querySelector("[data-live-events]");
  var tape = document.querySelector("[data-live-tape]");
  var scaleRows = {};
  SST.qsa("[data-scale-class]", root).forEach(function (li) {
    scaleRows[li.getAttribute("data-scale-class")] = {
      fill: li.querySelector("[data-scale-fill]"),
      count: li.querySelector("[data-scale-count]")
    };
  });

  var session = null;          // { id, … } from POST /api/live/sessions
  var stream = null;           // MediaStream
  var audioContext = null;
  var muted = false;
  var running = false;
  var seq = 0;
  var windowsSeen = 0;
  var confirmedEvents = 0;
  var classCounts = {};
  var sessionsPanel = document.querySelector("[data-live-sessions]");

  var windowSeconds = SST.threshold("audio.live_window_sec", 2) || 2;
  var sampleRate = 16000;

  function onTeardown(fn) {
    if (SST.lifecycle && SST.lifecycle.onTeardown) {
      SST.lifecycle.onTeardown(fn);
    }
  }

  // btoa of a large window must be built in chunks; String.fromCharCode.apply
  // has an argument-count ceiling, hence the 32 KiB slices.
  function base64FromBytes(bytes) {
    var binary = "";
    var CHUNK = 0x8000;
    for (var i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return window.btoa(binary);
  }

  /* --------------------------------------------------------------- state -- */

  function setState(text) {
    if (stateLabel) { stateLabel.textContent = text; }
  }

  function setButtons() {
    if (startBtn) { startBtn.disabled = running || !session; }
    if (stopBtn) { stopBtn.disabled = !running; }
    if (muteBtn) { muteBtn.disabled = !running; }
  }

  function fmtConf(v) {
    return (v == null) ? "—" : Number(v).toFixed(3);
  }

  /* ------------------------------------------------------------- consent -- */

  if (consentCheck && consentGrant) {
    consentCheck.addEventListener("change", function () {
      consentGrant.disabled = !consentCheck.checked;
    });
  }
  if (consentGrant) {
    consentGrant.addEventListener("click", function () {
      consentGrant.disabled = true;
      createSession();
    });
  }

  function createSession() {
    if (!analysisReady) {
      SST.toast("Install both model artifacts and restart the service before monitoring.");
      return;
    }
    setState("Opening session…");
    SST.api("/api/live/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent_ack: true,
        device_label: "browser",
        location: null
      })
    }).then(function (payload) {
      session = (payload && payload.data) || payload;
      seq = 0;
      if (sessionLabel) { sessionLabel.textContent = session.id ? session.id.slice(0, 8) : "—"; }
      setState("Ready");
      openMic();
    }).catch(function (error) {
      setState("Failed");
      SST.toast((error && error.message) || "Could not open a live session.");
      if (consentGrant) { consentGrant.disabled = !consentCheck.checked; }
    });
  }

  /* ------------------------------------------------------------ capture -- */

  function recordWindow() {
    return new Promise(function (resolve, reject) {
      var frames = [];
      var ctx = audioContext;
      var source = ctx.createMediaStreamSource(stream);
      var proc = ctx.createScriptProcessor(4096, 1, 1);
      source.connect(proc);
      // ScriptProcessor needs a destination to fire in some browsers; mute it.
      var silent = ctx.createGain();
      silent.gain.value = 0;
      proc.connect(silent);
      silent.connect(ctx.destination);

      var startedAt = performance.now();
      proc.onaudioprocess = function (e) {
        var input = e.inputBuffer.getChannelData(0);
        frames.push(new Float32Array(input));
        if (meterFill) {
          var peak = 0;
          for (var i = 0; i < input.length; i++) {
            var v = Math.abs(input[i]);
            if (v > peak) { peak = v; }
          }
          var db = 20 * Math.log10(Math.max(peak, 1e-10));
          meterFill.style.width = Math.max(0, Math.min(100, 100 + db)) + "%";
          if (meterPeak) { meterPeak.textContent = db.toFixed(0) + " dB"; }
        }
      };
      setTimeout(function () {
        proc.onaudioprocess = null;
        try { proc.disconnect(); source.disconnect(); } catch (err) { /* already gone */ }
        var total = 0;
        frames.forEach(function (f) { total += f.length; });
        var samples = new Float32Array(total);
        var at = 0;
        frames.forEach(function (f) { samples.set(f, at); at += f.length; });
        var duration = (performance.now() - startedAt) / 1000;
        if (samples.length < ctx.sampleRate * 0.5) {
          reject(new Error("Captured window was too short."));
          return;
        }
        resolve({
          samples: samples,
          sampleRate: ctx.sampleRate,
          duration: duration,
          capturedAt: new Date().toISOString()
        });
      }, windowSeconds * 1000);
    });
  }

  function encodeWav(samples, rate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function writeStr(off, s) {
      for (var i = 0; i < s.length; i++) { view.setUint8(off + i, s.charCodeAt(i)); }
    }
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, samples.length * 2, true);
    var off = 44;
    for (var i = 0; i < samples.length; i++, off += 2) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Uint8Array(buffer);
  }

  /* -------------------------------------------------------------- round -- */

  function pushWindow() {
    if (!running || !session) { return; }
    var next = recordWindow();
    next.then(function (capture) {
      seq += 1;
      var body = {
        seq: seq,
        audio_b64: toBase64(capture.samples, capture.sampleRate),
        sample_rate: capture.sampleRate,
        captured_at: capture.capturedAt,
        duration_sec: capture.samples.length / capture.sampleRate,
        duration_seconds: capture.samples.length / capture.sampleRate
      };
      return SST.api("/api/live/sessions/" + session.id + "/windows", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }).then(function (payload) {
        renderWindow((payload && payload.data) || {});
      }).catch(function (error) {
        seq -= 1; // this window never landed; let the next one take the number.
        renderWindowError(error);
      });
    }).catch(function (error) {
      setState("Capture failed");
      SST.toast((error && error.message) || "Capture failed.");
    }).then(function () {
      if (running) { scheduleNext(); }
    });
  }

  function scheduleNext() {
    if (!running) { return; }
    window.setTimeout(pushWindow, 400);
  }

  function toBase64(samples, rate) {
    // The server accepts a WAV container (RIFF) through the same decode path
    // as an upload, so a live window and an uploaded clip are analysed identically.
    return base64FromBytes(encodeWav(samples, rate));
  }

  /* -------------------------------------------------------------- render -- */

  function setCard(kind, cls, conf, note) {
    var card = document.querySelector('[data-model-card="' + kind + '"]');
    if (!card) { return; }
    var c = card.querySelector("[data-model-class]");
    var b = card.querySelector("[data-model-confidence]");
    var bar = card.querySelector("[data-model-bar]");
    var n = card.querySelector("[data-model-note]");
    if (c) { c.textContent = cls || "—"; }
    if (b) { b.textContent = conf == null ? "—" : fmtConf(conf); }
    if (bar) { bar.style.width = conf == null ? "0%" : Math.round(conf * 100) + "%"; }
    if (n) { n.textContent = note || ""; }
  }

  function bumpScale(cls) {
    if (!cls) { return; }
    classCounts[cls] = (classCounts[cls] || 0) + 1;
    var row = scaleRows[cls];
    if (!row) { return; }
    var max = 1;
    Object.keys(classCounts).forEach(function (k) {
      if (classCounts[k] > max) { max = classCounts[k]; }
    });
    if (row.fill) { row.fill.style.width = Math.round((classCounts[cls] / max) * 100) + "%"; }
    if (row.count) { row.count.textContent = classCounts[cls]; }
  }

  function renderWindowError(error) {
    if (verdictConsistency) {
      verdictConsistency.textContent = (error && error.message) || "Window failed.";
    }
    if (error && error.code === "pipeline_unavailable") {
      setState("Pipeline unavailable");
      running = false;
      setButtons();
    }
  }

  function renderWindow(w) {
    windowsSeen += 1;
    if (windowsCount) { windowsCount.textContent = String(windowsSeen); }

    var cls = (w.prediction && w.prediction.class) || "—";
    var conf = w.prediction && w.prediction.confidence;
    var gtm = w.gtm || {};

    setCard("python", cls, conf,
      w.latency_ms != null ? ("Inference " + w.latency_ms + " ms") : "");
    setCard("gtm", gtm.class, gtm.confidence, "");

    if (verdictClass) {
      verdictClass.textContent = w.confirmed ? (cls + " — confirmed") : cls;
    }
    if (verdictConsistency) {
      verdictConsistency.textContent = w.consistency_status ||
        ((w.quality && ("quality: " + w.quality)) || "");
    }
    if (differenceValue) {
      differenceValue.textContent = w.confidence_difference == null
        ? "—" : "|" + Number(w.confidence_difference).toFixed(3) + "|";
    }

    bumpScale(cls);
    if (w.confirmed) {
      confirmedEvents += 1;
      if (eventsCount) { eventsCount.textContent = String(confirmedEvents); }
    }

    if (tape) {
      var empty = tape.querySelector(".tape__item--empty");
      if (empty) { empty.remove(); }
      var li = el("li", { "class": "tape__item" }, [
        el("span", { "class": "tape__seq", text: "#" + w.seq }),
        el("span", { "class": "tape__class", text: cls }),
        el("span", { "class": "tape__conf", text: fmtConf(conf) }),
        el("span", { "class": "tape__status", text: w.consistency_status || "" }),
        el("span", { "class": "tape__ms", text: w.latency_ms != null ? w.latency_ms + " ms" : "" })
      ]);
      tape.appendChild(li);
      while (tape.children.length > 50) { tape.removeChild(tape.firstChild); }
    }
  }

  /* --------------------------------------------------------- start/stop -- */

  function start() {
    if (running || !session) { return; }
    running = true;
    setButtons();
    setState("Monitoring");
    pushWindow();
  }

  function stop() {
    running = false;
    setButtons();
    setState("Idle");
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    if (audioContext) {
      audioContext.close().catch(function () {});
      audioContext = null;
    }
    var sid = session && session.id;
    if (sid) {
      SST.api("/api/live/sessions/" + sid + "/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      }).then(refreshSessions).catch(function () { refreshSessions(); });
    }
    session = null;
    if (sessionLabel) { sessionLabel.textContent = "no session"; }
    if (consentCheck) { consentCheck.checked = false; }
    if (consentGrant) { consentGrant.disabled = true; }
    if (consentPanel) { consentPanel.hidden = false; }
  }

  if (muteBtn) {
    muteBtn.addEventListener("click", function () {
      muted = !muted;
      muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
      if (stream) {
        stream.getAudioTracks().forEach(function (t) { t.enabled = !muted; });
      }
      muteBtn.textContent = muted ? "Unmute" : "Mute";
    });
  }
  if (startBtn) { startBtn.addEventListener("click", start); }
  if (stopBtn) { stopBtn.addEventListener("click", stop); }

  function openMic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setState("Unsupported browser");
      return;
    }
    setState("Requesting microphone…");
    navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    }).then(function (mediaStream) {
      stream = mediaStream;
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
      setState("Ready");
      setButtons();
      if (consentPanel) { consentPanel.hidden = true; }
      start();
    }).catch(function (error) {
      setState("Microphone refused");
      SST.toast((error && (error.name === "NotAllowedError"
        ? "Microphone permission was refused."
        : error.message)) || "Microphone unavailable.");
      session = null;
      if (sessionLabel) { sessionLabel.textContent = "no session"; }
      if (consentPanel) { consentPanel.hidden = false; }
      if (consentGrant) { consentGrant.disabled = !consentCheck.checked; }
    });
  }

  onTeardown(stop);

  function refreshSessions() {
    if (!sessionsPanel) { return; }
    SST.api("/api/live/sessions").then(function (payload) {
      var rows = (payload && payload.data) || [];
      sessionsPanel.textContent = "";
      if (!rows.length) {
        sessionsPanel.appendChild(el("p", { "class": "prose", text: "No microphone sessions yet." }));
        return;
      }
      var list = el("ol", { "class": "session-list" });
      rows.forEach(function (row) {
        var item = el("li", { "class": "session-list__item" });
        item.appendChild(el("span", { "class": "session-list__date",
          text: row.started_at ? new Date(row.started_at).toLocaleString() : "Unknown time" }));
        item.appendChild(el("span", { "class": "session-list__meta",
          text: row.status + " · " + row.window_count + " windows · " + row.alert_count + " alerts" }));
        list.appendChild(item);
      });
      sessionsPanel.appendChild(list);
    }).catch(function (error) {
      sessionsPanel.textContent = "";
      sessionsPanel.appendChild(el("p", { "class": "prose",
        text: (error && error.message) || "Session history could not be loaded." }));
    });
  }

  // A reload that lands back on this page with an active session cannot be
  // resumed (the streak state lives server-side); show the consent gate again.
  setButtons();
  if (!analysisReady) { setState("Models unavailable"); }
  refreshSessions();
})(window.SST = window.SST || {});
