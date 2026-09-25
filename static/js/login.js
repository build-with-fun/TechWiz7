/* Keep the native form POST; this script adds inline validation and submit feedback. */
"use strict";

(function () {
  var form = document.querySelector("[data-login-form]");
  var submit = document.querySelector("[data-login-submit]");
  if (!form || !submit) {
    return;
  }

  form.addEventListener("submit", function (event) {
    var username = form.querySelector("#username");
    var password = form.querySelector("#password");

    // The form opts out of native validation so the missing field can receive focus.
    var missing = [];
    if (username && !username.value.trim()) {
      missing.push(username);
    }
    if (password && !password.value) {
      missing.push(password);
    }
    if (missing.length > 0) {
      event.preventDefault();
      missing.forEach(function (field) {
        field.setAttribute("aria-invalid", "true");
      });
      if (missing[0]) {
        missing[0].focus();
      }
      return;
    }

    if (submit.dataset.busy === "1") {
      event.preventDefault();
      return;
    }
    submit.dataset.busy = "1";
    submit.disabled = true;
    var original = submit.textContent;
    submit.textContent = "Signing in…";
    // Restore the control if the navigation never completes (for example offline).
    window.setTimeout(function () {
      submit.dataset.busy = "";
      submit.disabled = false;
      submit.textContent = original;
    }, 15000);
  });
})();
