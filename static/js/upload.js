/**
 * upload.js — the drag-and-drop upload queue (progressive enhancement).
 *
 * Without JavaScript the file input still posts the form normally. With
 * JavaScript each chosen file is POSTed to /api/audio/upload with SST.api,
 * and the queue shows the outcome: the event (both models' predictions and
 * the consistency verdict), the 409 duplicate (with a one-click re-send that
 * allows duplicates), a 422 unusable-audio refusal, or a network failure.
 *
 * DOM contract (from upload.html):
 *   [data-dropzone]        the drop target
 *   [data-dropzone-input]  the file input, labelled by its visible picker
 *   [data-upload-queue]    the <ol> the queue items render into
 */
"use strict";

(function (SST) {
  var dropzone = document.querySelector("[data-dropzone]");
  var input = document.querySelector("[data-dropzone-input]");
  var queue = document.querySelector("[data-upload-queue]");
  if (!dropzone || !input || !queue || typeof SST === "undefined") {
    return; // progressive enhancement: plain form POST still works.
  }
  input.multiple = true;

  /* ------------------------------------------------------------- queue ui - */

  function addItem(name) {
    var li = SST.el("li", { "class": "queue__item", "data-upload-item": name });
    li.appendChild(SST.el("p", { "class": "queue__name" }, name));
    var body = SST.el("div", { "class": "queue__body" });
    li.appendChild(body);
    queue.prepend(li);
    return { li: li, region: SST.region(body) };
  }

  /* ------------------------------------------------------------- requests - */

  function send(file, allowDuplicate, slot) {
    slot.region.loading("Uploading and analysing…", 3);

    var form = new FormData();
    form.append("file", file, file.name);
    form.append("filename", file.name);
    form.append("source", "upload");
    if (allowDuplicate) {
      form.append("allow_duplicate", "true");
    }

    return SST.api("/api/audio/upload", { method: "POST", body: form })
      .then(function (payload) {
        var event = (payload && payload.data) || payload || {};
        var cls = event.effective_class || event.predicted_class || "—";
        var comparison = event.comparison || {};
        var rows = SST.el("div", { "class": "stack stack--tight" });
        rows.appendChild(SST.el("p", { "class": "queue__ok" },
          "Classified as " + cls + "."));

        var meta = [];
        var py = event.python || {};
        var tm = event.gtm || {};
        if (py.predicted_class) {
          meta.push("Python: " + py.predicted_class +
            (py.confidence != null ? " (" + Number(py.confidence).toFixed(3) + ")" : ""));
        }
        if (tm.predicted_class) {
          meta.push("GTM: " + tm.predicted_class +
            (tm.confidence != null ? " (" + Number(tm.confidence).toFixed(3) + ")" : ""));
        }
        if (comparison.consistency_verdict) {
          meta.push("Consistency: " + comparison.consistency_verdict);
        }
        if (meta.length) {
          rows.appendChild(SST.el("p", { "class": "queue__note" }, meta.join(" · ")));
        }

        var eventId = event.event_id || event.id;
        var href = eventId ? SST.endpointFor("eventDetail", { eventId: eventId }) : null;
        rows.appendChild(SST.el("p", { "class": "queue__actions" }, [
          href ? SST.el("a", { "class": "btn btn--ghost btn--sm", href: href }, "Open event")
               : SST.el("a", { "class": "btn btn--ghost btn--sm", href: SST.endpoint("events") }, "All events")
        ]));
        slot.region.render(rows);
      })
      .catch(function (error) {
        if (error && error.code === "duplicate_audio") {
          var detail = (error.details && error.details.audio_id) || "";
          var box = SST.el("div", { "class": "stack stack--tight" });
          box.appendChild(SST.el("p", { "class": "queue__warn" },
            "Duplicate: these bytes are already stored as " + detail + "."));
          var again = SST.el("button", { "class": "btn btn--ghost btn--sm", type: "button",
                                        text: "Analyse again anyway" });
          again.addEventListener("click", function () {
            again.disabled = true;
            send(file, true, slot);
          });
          box.appendChild(again);
          slot.region.render(box);
          return;
        }
        var message = (error && (error.message || error.code)) || "Upload failed.";
        slot.region.error({ title: "Upload failed", message: message,
                            requestId: error && error.requestId });
      });
  }

  /* ------------------------------------------------------------ file entry - */

  function handleFiles(files) {
    Array.prototype.forEach.call(files || [], function (file) {
      var slot = addItem(file.name);
      send(file, false, slot);
    });
  }

  dropzone.addEventListener("submit", function (event) {
    event.preventDefault();
    handleFiles(input.files);
    input.value = "";
  });
  input.addEventListener("change", function () {
    handleFiles(input.files);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.setAttribute("data-drag-active", "1");
    });
  });
  ["dragleave", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.removeAttribute("data-drag-active");
    });
  });
  dropzone.addEventListener("drop", function (event) {
    if (event.dataTransfer && event.dataTransfer.files) {
      handleFiles(event.dataTransfer.files);
    }
  });
})(window.SST);
