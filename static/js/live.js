/**
 * Live microphone monitor (FR vi, vii, lxiv, lxxix).
 *
 * The microphone is read continuously into a buffer and a timer cuts a window every
 * live_window_sec. Windows wait in a small queue and go to the server one at a time.
 * (The first version recorded, sent and waited in turn, so anything that happened
 * during inference was missed.) If the server falls behind, the oldest window is
 * dropped and the count shown on screen.
 *
 * Microphone states (FR vii): Available, Active, Paused, Disconnected, Permission
 * denied, plus "Not available" when there is no microphone.
 *
 * Nothing is captured before consent is given. While capturing, the tab title starts
 * with a red dot, and Stop releases the device.
 */
"use strict";

(function (SST) {
  var root = document.querySelector("[data-live-root]");
  if (!root || typeof SST === "undefined") {
    return;
  }

  var el = SST.el;
  var analysisReady = root.getAttribute("data-analysis-ready") === "1";
  var q = function (sel) { return document.querySelector(sel); };

  var consentPanel = q("[data-consent-panel]");
  var consentCheck = q("[data-consent-check]");
  var consentGrant = q("[data-consent-grant]");
  var stateLabel = q("[data-live-state]");
  var micStatus = q("[data-mic-status]");
  var sessionLabel = q("[data-live-session-id]");
  var startBtn = q("[data-live-start]");
  var stopBtn = q("[data-live-stop]");
  var pauseBtn = q("[data-live-pause]");
  var meterFill = q("[data-live-meter-fill]");
  var meterPeak = q("[data-live-meter-peak]");
  var verdictClass = q("[data-live-class]");
  var verdictConsistency = q("[data-live-consistency]");
  var differenceValue = q("[data-live-difference]");
  var windowsCount = q("[data-live-windows]");
  var eventsCount = q("[data-live-events]");
  var droppedCount = q("[data-live-dropped]");
  var alertBox = q("[data-live-alert]");
  var tape = q("[data-live-tape]");
  var sessionsPanel = q("[data-live-sessions]");
  var scaleRows = {};
  SST.qsa("[data-scale-class]", root).forEach(function (li) {
    scaleRows[li.getAttribute("data-scale-class")] = {
      fill: li.querySelector("[data-scale-fill]"),
      count: li.querySelector("[data-scale-count]")
    };
  });

  var MAX_QUEUE = 2;
  var windowSeconds = SST.threshold("audio.live_window_sec", 2) || 2;
  var sampleRate = 16000;
  var originalTitle = document.title;

  var session = null;       // { id, ... } from POST /api/live/sessions
  var stream = null;        // MediaStream
  var audioContext = null;
  var processor = null;
  var source = null;
  var ring = [];            // Float32Array chunks newer than the last cut
  var ringLength = 0;
  var cutTimer = null;
  var queue = [];           // encoded windows waiting to be sent
  var inFlight = null;      // AbortController of the request being sent
  var running = false;
  var paused = false;
  var seq = 0;
  var counts = { windows: 0, events: 0, dropped: 0 };
  var classCounts = {};

  function onTeardown(fn) {
    if (SST.lifecycle && SST.lifecycle.onTeardown) {
      SST.lifecycle.onTeardown(fn);
    }
  }

  // Status

  var MIC_TONES = {
    "Available": "ok", "Active": "live", "Paused": "warn",
    "Disconnected": "bad", "Permission denied": "bad", "Not available": "muted"
  };

  function setMic(status) {
    if (micStatus) {
      micStatus.textContent = status;
      micStatus.className = "pip mic-status pip--" + (MIC_TONES[status] || "muted");
      micStatus.setAttribute("data-mic", status);
    }
    document.title = status === "Active" ? "● " + originalTitle : originalTitle;
  }

  function setState(text) {
    if (stateLabel) { stateLabel.textContent = text; }
  }

  function setButtons() {
    if (startBtn) { startBtn.disabled = running || !session; }
    if (stopBtn) { stopBtn.disabled = !running; }
    if (pauseBtn) {
      pauseBtn.disabled = !running;
      pauseBtn.setAttribute("aria-pressed", paused ? "true" : "false");
      var label = pauseBtn.querySelector("span");
      if (label) { label.textContent = paused ? "Resume" : "Pause"; }
    }
  }

  function setCount(node, value) {
    if (node) { node.textContent = String(value); }
  }

  // Check for a microphone and an earlier "no" without opening the device.
  function probeMicrophone() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMic("Not available");
      return;
    }
    var devices = navigator.mediaDevices.enumerateDevices
      ? navigator.mediaDevices.enumerateDevices()
      : Promise.resolve([{ kind: "audioinput" }]);
    devices.then(function (list) {
      var hasInput = list.some(function (d) { return d.kind === "audioinput"; });
      if (!hasInput) { setMic("Not available"); return; }
      if (!navigator.permissions || !navigator.permissions.query) {
        setMic("Available");
        return;
      }
      navigator.permissions.query({ name: "microphone" }).then(function (perm) {
        var apply = function () {
          if (running) { return; }
          setMic(perm.state === "denied" ? "Permission denied" : "Available");
        };
        apply();
        perm.onchange = apply;
      }).catch(function () { setMic("Available"); });
    }).catch(function () { setMic("Available"); });
  }

  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", function () {
      if (!running) { probeMicrophone(); }
    });
  }

  // Consent

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
    setState("Opening session...");
    SST.api("/api/live/sessions", {
      method: "POST",
      json: { consent_ack: true, device_label: "browser", location: null }
    }).then(function (payload) {
      session = (payload && payload.data) || payload;
      seq = 0;
      if (sessionLabel) { sessionLabel.textContent = session.id ? session.id.slice(0, 8) : "-"; }
      openMic();
    }).catch(function (error) {
      setState("Failed");
      SST.toast((error && error.message) || "Could not open a live session.");
      if (consentGrant) { consentGrant.disabled = !consentCheck.checked; }
    });
  }

  function openMic() {
    setState("Requesting microphone...");
    navigator.mediaDevices.getUserMedia({
      // Browser "enhancements" are switched off: echo cancellation and noise
      // suppression are tuned for speech and flatten gunshots and alarms.
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    }).then(function (mediaStream) {
      stream = mediaStream;
      stream.getAudioTracks().forEach(function (track) {
        track.addEventListener("ended", onDisconnected);
      });
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
      if (consentPanel) { consentPanel.hidden = true; }
      start();
    }).catch(function (error) {
      var denied = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
      setMic(denied ? "Permission denied" : "Not available");
      setState(denied ? "Permission denied" : "Microphone unavailable");
      SST.toast(denied
        ? "Microphone permission was refused. Allow it in the browser's site settings to monitor."
        : ((error && error.message) || "No usable microphone was found."));
      closeSession();
    });
  }

  // Capture

  function startCapture() {
    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    // A ScriptProcessor only runs when connected to the output; a zero-gain node
    // keeps it running without playing the microphone back through the speakers.
    var silent = audioContext.createGain();
    silent.gain.value = 0;
    source.connect(processor);
    processor.connect(silent);
    silent.connect(audioContext.destination);
    processor.onaudioprocess = function (e) {
      var input = e.inputBuffer.getChannelData(0);
      updateMeter(input);
      if (paused) { return; }
      ring.push(new Float32Array(input));
      ringLength += input.length;
    };
    cutTimer = window.setInterval(cutWindow, windowSeconds * 1000);
  }

  function stopCapture() {
    if (cutTimer) { window.clearInterval(cutTimer); cutTimer = null; }
    if (processor) {
      processor.onaudioprocess = null;
      try { processor.disconnect(); } catch (err) { /* already gone */ }
      processor = null;
    }
    if (source) {
      try { source.disconnect(); } catch (err) { /* already gone */ }
      source = null;
    }
    ring = [];
    ringLength = 0;
  }

  function updateMeter(input) {
    if (!meterFill) { return; }
    var peak = 0;
    for (var i = 0; i < input.length; i++) {
      var v = Math.abs(input[i]);
      if (v > peak) { peak = v; }
    }
    var db = 20 * Math.log10(Math.max(peak, 1e-10));
    meterFill.style.width = Math.max(0, Math.min(100, 100 + db)) + "%";
    if (meterPeak) { meterPeak.textContent = db.toFixed(0) + " dB"; }
  }

  function cutWindow() {
    if (!running || paused || !audioContext) { return; }
    var rate = audioContext.sampleRate;
    if (ringLength < rate * 0.5) { return; }   // less than half a second: skip, not an error
    var samples = new Float32Array(ringLength);
    var at = 0;
    ring.forEach(function (chunk) { samples.set(chunk, at); at += chunk.length; });
    ring = [];
    ringLength = 0;
    seq += 1;
    enqueue({
      seq: seq,
      audio_b64: base64FromBytes(encodeWav(samples, rate)),
      sample_rate: rate,
      captured_at: new Date().toISOString(),
      duration_sec: samples.length / rate
    });
  }

  function enqueue(body) {
    queue.push(body);
    while (queue.length > MAX_QUEUE) {
      queue.shift();
      counts.dropped += 1;
      setCount(droppedCount, counts.dropped);
    }
    sendNext();
  }

  function sendNext() {
    if (inFlight || !queue.length || !session) { return; }
    var body = queue.shift();
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    inFlight = controller || {};
    SST.api("/api/live/sessions/" + session.id + "/windows", {
      method: "POST",
      json: body,
      signal: controller ? controller.signal : undefined
    }).then(function (payload) {
      renderWindow((payload && payload.data) || {});
    }).catch(function (error) {
      if (error && error.name === "AbortError") { return; }
      renderWindowError(error);
    }).then(function () {
      inFlight = null;
      if (running) { sendNext(); }
    });
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

  // 16-bit PCM WAV, so a live window goes through the same decoder as an upload.
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

  // Render

  function fmtConf(v) {
    return (v == null) ? "-" : Number(v).toFixed(3);
  }

  function setCard(kind, cls, conf, top3, note) {
    var card = q('[data-model-card="' + kind + '"]');
    if (!card) { return; }
    var c = card.querySelector("[data-model-class]");
    var b = card.querySelector("[data-model-confidence]");
    var bar = card.querySelector("[data-model-bar]");
    var list = card.querySelector("[data-model-top3]");
    var n = card.querySelector("[data-model-note]");
    if (c) { c.textContent = cls || "-"; }
    if (b) { b.textContent = conf == null ? "-" : fmtConf(conf); }
    if (bar) { bar.style.width = conf == null ? "0%" : Math.round(conf * 100) + "%"; }
    if (list) {
      list.textContent = "";
      (top3 || []).forEach(function (item) {
        list.appendChild(el("li", { text: item["class"] + " " + fmtConf(item.confidence) }));
      });
    }
    if (n) { n.textContent = note || ""; }
  }

  function bumpScale(cls) {
    if (!cls) { return; }
    classCounts[cls] = (classCounts[cls] || 0) + 1;
    var max = 1;
    Object.keys(classCounts).forEach(function (k) {
      if (classCounts[k] > max) { max = classCounts[k]; }
    });
    Object.keys(scaleRows).forEach(function (k) {
      var row = scaleRows[k];
      var count = classCounts[k] || 0;
      if (row.fill) { row.fill.style.width = Math.round((count / max) * 100) + "%"; }
      if (row.count) { row.count.textContent = count; }
    });
  }

  function showAlert(w) {
    if (!alertBox) { return; }
    var cls = w.prediction && w.prediction["class"];
    alertBox.hidden = false;
    alertBox.textContent = "";
    alertBox.appendChild(el("strong", { text: (w.severity_display || "Alert") + ": " + cls }));
    alertBox.appendChild(el("span", {
      text: " confirmed in " + w.consecutive + " consecutive windows. " +
        (w.recommended_action || "Review the audio before acting.")
    }));
    if (w.event_id) {
      alertBox.appendChild(el("a", { href: "/events/" + w.event_id, text: " Open event" }));
    }
  }

  function renderWindowError(error) {
    if (verdictConsistency) {
      verdictConsistency.textContent = (error && error.message) || "Window failed.";
    }
    if (error && error.code === "pipeline_unavailable") {
      setState("Models unavailable");
      stop();
    }
  }

  function renderWindow(w) {
    counts.windows += 1;
    setCount(windowsCount, counts.windows);

    var cls = (w.prediction && w.prediction["class"]) || "-";
    var conf = w.prediction && w.prediction.confidence;
    var gtm = w.gtm || {};
    var top3 = w.top3 || {};

    setCard("python", cls, conf, top3.python,
      w.latency_ms != null ? ("Window analysed in " + w.latency_ms + " ms") : "");
    setCard("gtm", gtm["class"], gtm.confidence, top3.gtm, "");

    if (verdictClass) {
      verdictClass.textContent = w.confirmed ? (cls + " (confirmed)") : cls;
    }
    if (verdictConsistency) {
      var parts = [w.consistency_status, w.quality && ("quality " + w.quality)];
      if (w.review_required) { parts.push("manual review"); }
      verdictConsistency.textContent = parts.filter(Boolean).join(" · ");
    }
    if (differenceValue) {
      differenceValue.textContent = w.confidence_difference == null
        ? "-" : Number(w.confidence_difference).toFixed(3);
    }

    bumpScale(cls);
    if (w.alert) {
      counts.events += 1;
      setCount(eventsCount, counts.events);
      showAlert(w);
    }

    if (tape) {
      var empty = tape.querySelector(".tape__item--empty");
      if (empty) { empty.remove(); }
      tape.appendChild(el("li", { "class": "tape__item" + (w.alert ? " tape__item--alert" : "") }, [
        el("span", { "class": "tape__seq", text: "#" + w.seq }),
        el("span", { "class": "tape__class", text: cls }),
        el("span", { "class": "tape__conf", text: fmtConf(conf) }),
        el("span", { "class": "tape__status", text: w.consistency_status || "" }),
        el("span", { "class": "tape__ms", text: w.latency_ms != null ? w.latency_ms + " ms" : "" })
      ]));
      while (tape.children.length > 50) { tape.removeChild(tape.firstChild); }
    }
  }

  // Start / stop

  function start() {
    if (running || !session || !stream) { return; }
    running = true;
    paused = false;
    startCapture();
    setMic("Active");
    setState("Monitoring");
    setButtons();
  }

  function releaseDevice() {
    stopCapture();
    if (stream) {
      stream.getTracks().forEach(function (t) {
        t.removeEventListener("ended", onDisconnected);
        t.stop();
      });
      stream = null;
    }
    if (audioContext) {
      audioContext.close().catch(function () {});
      audioContext = null;
    }
    queue = [];
    if (inFlight && inFlight.abort) { inFlight.abort(); }
    inFlight = null;
  }

  function closeSession() {
    var sid = session && session.id;
    session = null;
    if (sid) {
      SST.api("/api/live/sessions/" + sid + "/stop", { method: "POST", json: {} })
        .then(refreshSessions).catch(refreshSessions);
    }
    if (sessionLabel) { sessionLabel.textContent = "no session"; }
    if (consentCheck) { consentCheck.checked = false; }
    if (consentGrant) { consentGrant.disabled = true; }
    if (consentPanel) { consentPanel.hidden = false; }
  }

  function stop() {
    var wasRunning = running;
    running = false;
    paused = false;
    releaseDevice();
    closeSession();
    setButtons();
    if (wasRunning) { setState("Stopped"); }
    probeMicrophone();
  }

  function onDisconnected() {
    // Device unplugged or taken by another app: end the session.
    running = false;
    releaseDevice();
    closeSession();
    setButtons();
    setMic("Disconnected");
    setState("Microphone disconnected");
    SST.toast("The microphone was disconnected. Monitoring has stopped.");
  }

  function togglePause() {
    if (!running) { return; }
    paused = !paused;
    if (paused) {
      ring = [];
      ringLength = 0;
    }
    setMic(paused ? "Paused" : "Active");
    setState(paused ? "Paused" : "Monitoring");
    setButtons();
  }

  if (pauseBtn) { pauseBtn.addEventListener("click", togglePause); }
  if (startBtn) { startBtn.addEventListener("click", start); }
  if (stopBtn) { stopBtn.addEventListener("click", stop); }
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

  // A reload cannot resume a session (the browser gives up the device), so the
  // page always starts at the consent gate.
  setButtons();
  if (!analysisReady) { setState("Models unavailable"); }
  probeMicrophone();
  refreshSessions();
})(window.SST = window.SST || {});
