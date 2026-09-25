/**
 * Event detail page: the audio player and its server-computed visuals.
 *
 * Everything here is presentation. The waveform peaks and the spectrogram come from
 * /api/events/<id>/visuals, which computes them from the SAME stored file the models
 * were given -- the browser never re-decodes the audio for drawing. The <audio>
 * element only supplies playback (current time, seeking, volume).
 */
(function () {
  "use strict";

  var playerEl = document.querySelector("[data-audio-player]");
  if (!playerEl || !window.SST) return;

  var src = playerEl.getAttribute("data-src");
  var visualsUrl = playerEl.getAttribute("data-visuals");

  var playBtn = playerEl.querySelector("[data-player-play]");
  var timeEl = playerEl.querySelector("[data-player-time]");
  var seekEl = playerEl.querySelector("[data-player-seek]");
  var volumeEl = playerEl.querySelector("[data-player-volume]");
  var durationEl = playerEl.querySelector("[data-player-duration]");
  var waveCanvas = playerEl.querySelector("[data-player-waveform]");
  var specCanvas = playerEl.querySelector("[data-player-spectrogram]");

  var audio = null;
  var visuals = null;
  var playIcon = playBtn ? playBtn.innerHTML : "";
  var pauseIcon = playBtn ? playBtn.innerHTML : "";

  function fmt(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    var m = Math.floor(sec / 60);
    var s = sec - m * 60;
    return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
  }

  /* ---------------------------------------------------------------- audio element */

  function ensureAudio() {
    if (audio) return audio;
    audio = new Audio(src);
    audio.preload = "metadata";
    audio.volume = volumeEl ? Number(volumeEl.value) / 100 : 0.8;
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("loadedmetadata", onMeta);
    audio.addEventListener("play", onPlayState);
    audio.addEventListener("pause", onPlayState);
    audio.addEventListener("ended", onPlayState);
    return audio;
  }

  function onMeta() {
    if (durationEl && isFinite(audio.duration)) {
      durationEl.textContent = fmt(audio.duration);
    }
    onTime();
  }

  function onTime() {
    if (!audio) return;
    if (timeEl) timeEl.textContent = fmt(audio.currentTime);
    if (seekEl && isFinite(audio.duration) && audio.duration > 0 && !seeking) {
      seekEl.value = String(Math.round((audio.currentTime / audio.duration) * 1000));
    }
    drawProgress();
  }

  function onPlayState() {
    if (!playBtn) return;
    playBtn.innerHTML = audio && !audio.paused ? pauseIcon : playIcon;
    playBtn.setAttribute("aria-label", audio && !audio.paused ? "Pause recording" : "Play recording");
  }

  var seeking = false;
  if (seekEl) {
    seekEl.addEventListener("input", function () {
      seeking = true;
      if (audio && isFinite(audio.duration)) {
        audio.currentTime = (Number(seekEl.value) / 1000) * audio.duration;
      }
      if (timeEl && audio) timeEl.textContent = fmt(audio.currentTime);
      drawProgress();
    });
    seekEl.addEventListener("change", function () { seeking = false; });
  }
  if (volumeEl) {
    volumeEl.addEventListener("input", function () {
      if (audio) audio.volume = Number(volumeEl.value) / 100;
    });
  }
  if (playBtn) {
    playBtn.addEventListener("click", function () {
      var a = ensureAudio();
      if (a.paused) {
        var p = a.play();
        if (p && p.catch) p.catch(function () { /* autoplay refusal: user clicked, so unlikely */ });
      } else {
        a.pause();
      }
    });
  }

  /* ---------------------------------------------------------------- canvas drawing */

  function fit(canvas) {
    if (!canvas) return null;
    var ratio = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || canvas.width;
    var cssH = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(cssW * ratio)) {
      canvas.width = Math.round(cssW * ratio);
      canvas.height = Math.round(cssH * ratio);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx: ctx, w: cssW, h: cssH };
  }

  function drawWaveform() {
    if (!waveCanvas || !visuals || !visuals.peaks || !visuals.peaks.length) return;
    var f = fit(waveCanvas);
    if (!f) return;
    var ctx = f.ctx, w = f.w, h = f.h;
    var css = getComputedStyle(waveCanvas);
    ctx.clearRect(0, 0, w, h);
    var stroke = css.color || "#38bdf8";
    var mid = h / 2;
    var peaks = visuals.peaks;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var x = 0; x < w; x++) {
      var p = peaks[Math.min(peaks.length - 1, Math.floor((x / w) * peaks.length))] || 0;
      var amp = Math.max(0.02, p) * (h / 2 - 2);
      ctx.moveTo(x + 0.5, mid - amp);
      ctx.lineTo(x + 0.5, mid + amp);
    }
    ctx.stroke();
    drawProgress();
  }

  function drawSpectrogram() {
    if (!specCanvas || !visuals || !visuals.spectrogram) return;
    var spec = visuals.spectrogram;
    var data = spec.data;
    var mels = spec.n_mels, frames = spec.n_frames;
    if (!data || !mels || !frames) return;
    var f = fit(specCanvas);
    if (!f) return;
    var ctx = f.ctx, w = f.w, h = f.h;
    var image = ctx.createImageData(Math.ceil(w), Math.ceil(h));
    var px = image.data;
    // Draw offscreen at native frame resolution then scale -- crisp and fast.
    var off = document.createElement("canvas");
    off.width = frames; off.height = mels;
    var octx = off.getContext("2d");
    var img = octx.createImageData(frames, mels);
    var d = img.data;
    for (var j = 0; j < mels; j++) {           // mel row 0 = low frequency = bottom
      for (var i = 0; i < frames; i++) {
        var v = data[j * frames + i];          // already normalised 0..1 server-side
        var idx = ((mels - 1 - j) * frames + i) * 4;
        // quiet -> deep navy, loud -> bright cyan (matches the page legend)
        d[idx] = Math.round(11 + 45 * v);
        d[idx + 1] = Math.round(18 + 169 * v);
        d[idx + 2] = Math.round(32 + 216 * v);
        d[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(off, 0, 0, frames, mels, 0, 0, w, h);
    void px; // image kept for potential per-pixel overlays later
  }

  function drawProgress() {
    if (!waveCanvas || !visuals || !audio || !isFinite(audio.duration) || audio.duration <= 0) return;
    var f = fit(waveCanvas);
    if (!f) return;
    var ctx = f.ctx, w = f.w, h = f.h;
    var frac = audio.currentTime / audio.duration;
    ctx.save();
    ctx.fillStyle = "rgba(56, 189, 248, 0.18)";
    ctx.fillRect(0, 0, w * frac, h);
    ctx.restore();
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    if (!visuals) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { drawWaveform(); drawSpectrogram(); }, 150);
  });

  /* ---------------------------------------------------------------- load visuals */

  SST.api(visualsUrl)
    .then(function (data) {
      visuals = data && data.data ? data.data : data;
      drawWaveform();
      drawSpectrogram();
    })
    .catch(function (error) {
      // Missing visuals must not break playback -- the player still works.
      if (waveCanvas) {
        var f = fit(waveCanvas);
        if (f) {
          f.ctx.clearRect(0, 0, f.w, f.h);
          f.ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
          f.ctx.font = "12px system-ui, sans-serif";
          f.ctx.fillText("Visuals unavailable (" + (error.code || "error") + ") — playback still works.", 8, f.h / 2);
        }
      }
    });

  SST.lifecycle.onTeardown(function () {
    if (audio) {
      audio.pause();
      audio.src = "";
      audio = null;
    }
  });
})();
