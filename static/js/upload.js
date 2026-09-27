/**
 * Drag-and-drop upload queue. Without JavaScript the form still posts normally.
 * Each file goes to /api/audio/upload and its row shows the result: the event, a 409
 * duplicate (with a button to send it anyway), a 422 refusal or a network error.
 *
 * Elements (upload.html): [data-dropzone], [data-dropzone-input], [data-upload-queue]
 */
"use strict";

(function (SST) {
  var dropzone = document.querySelector("[data-dropzone]");
  var input = document.querySelector("[data-dropzone-input]");
  var queue = document.querySelector("[data-upload-queue]");
  if (!dropzone || !input || !queue || typeof SST === "undefined") {
    return; // the plain form POST still works
  }
  input.multiple = true;

  // Queue UI

  function addItem(name) {
    var empty = queue.querySelector(".queue__item--empty");
    if (empty) empty.remove();
    var li = SST.el("li", { "class": "queue__item", "data-upload-item": name });
    li.appendChild(SST.el("p", { "class": "queue__name" }, name));
    var body = SST.el("div", { "class": "queue__body" });
    li.appendChild(body);
    queue.prepend(li);
    return { li: li, region: SST.region(body) };
  }

  // Result card

  function chip(kind, value) {
    if (!value) return null;
    var p = SST.presentation(kind, value);
    var prefix = { severity: "sev", quality: "quality", consistency: "consistency" }[kind];
    return SST.el("span", { "class": "chip chip--" + prefix + "-" + p.slug }, [
      SST.el("span", { "class": "chip__glyph", "aria-hidden": "true" }, p.glyph),
      SST.el("span", { "class": "chip__text" }, value)
    ]);
  }

  function topThree(confidences) {
    return Object.keys(confidences || {})
      .map(function (name) { return { name: name, value: Number(confidences[name]) || 0 }; })
      .sort(function (a, b) { return b.value - a.value; })
      .slice(0, 3);
  }

  function modelColumn(label, key, block) {
    var col = SST.el("div", { "class": "result__model result__model--" + key });
    col.appendChild(SST.el("p", { "class": "result__model-name" }, [
      SST.el("span", { "class": "dot dot--" + key, "aria-hidden": "true" }), label
    ]));
    var rows = topThree(block && block.confidences);
    if (!rows.length) {
      col.appendChild(SST.el("p", { "class": "muted" }, "No prediction returned."));
      return col;
    }
    var list = SST.el("ol", { "class": "result__scores" });
    rows.forEach(function (row, i) {
      var pct = Math.max(0, Math.min(100, row.value * 100));
      var bar = SST.el("span", { "class": "result__bar" }, [SST.el("span", { "class": "result__fill" })]);
      bar.firstChild.setAttribute("data-bar-width", pct.toFixed(1));
      list.appendChild(SST.el("li", { "class": "result__row" + (i === 0 ? " result__row--top" : "") }, [
        SST.el("span", { "class": "result__class" }, row.name), bar,
        SST.el("span", { "class": "result__pct" }, pct.toFixed(1) + "%")
      ]));
    });
    col.appendChild(list);
    return col;
  }

  function reasonText(ids) {
    var phrases = {};
    ((SST.vocabulary || {}).reviewConditions || []).forEach(function (c) { phrases[c.id] = c.srs_phrase || c.id; });
    return String(ids).split(",").map(function (id) {
      id = id.trim();
      return phrases[id] || id.replace(/_/g, " ");
    }).join(", ");
  }

  function resultCard(event) {
    var models = event.models || {};
    var card = SST.el("div", { "class": "result" });
    var head = SST.el("div", { "class": "result__head" });
    head.appendChild(SST.el("div", {}, [
      SST.el("span", { "class": "result__label" }, "Final decision"),
      SST.el("span", { "class": "result__class-name" }, event.predicted_class || "Undecided")
    ]));
    var chips = SST.el("div", { "class": "chips" });
    [chip("severity", event.severity), chip("quality", (event.quality || {}).verdict),
     chip("consistency", event.consistency_status)].forEach(function (c) { if (c) chips.appendChild(c); });
    if (event.alert) chips.appendChild(SST.el("span", { "class": "tag tag--alert" }, "Alert raised"));
    head.appendChild(chips);
    card.appendChild(head);

    card.appendChild(SST.el("div", { "class": "result__models" }, [
      modelColumn("Python model", "python", models.python),
      modelColumn("Teachable Machine", "gtm", models.gtm)
    ]));

    var facts = [];
    if (event.confidence_difference != null) {
      facts.push("Confidence difference " + Number(event.confidence_difference).toFixed(3));
    }
    if ((event.timing || {}).elapsed_ms != null) {
      facts.push("analysed in " + SST.format.ms(event.timing.elapsed_ms));
    }
    if (facts.length) card.appendChild(SST.el("p", { "class": "result__facts" }, facts.join(" · ")));
    if (event.requires_manual_review) {
      card.appendChild(SST.el("p", { "class": "result__review" },
        "Manual Review Required" + (event.review_reason ? ": " + reasonText(event.review_reason) : ".")));
    }
    var eventId = event.id || event.event_id;
    var href = eventId ? SST.endpointFor("eventDetail", { eventId: eventId }) : SST.endpoint("events");
    card.appendChild(SST.el("p", { "class": "queue__actions" }, [
      SST.el("a", { "class": "btn btn--sm", href: href }, eventId ? "Open event " + eventId : "All events")
    ]));
    SST.initBars(card);
    return card;
  }

  // Requests

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
        slot.region.render(resultCard((payload && payload.data) || payload || {}));
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

  // File entry

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
