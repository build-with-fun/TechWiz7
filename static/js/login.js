/**
 * login.js — the login page's progressive enhancement.
 *
 * Without JavaScript the form is a normal POST to /login. With JavaScript we
 * still do the normal POST (it is more robust than XHR for a login), but we
 * validate inline first and disable the submit button for the round trip so a
 * double-click cannot submit twice.
 *
 * DOM contract (from auth/login.html):
 *   [data-login-form]     the <form>
 *   [data-login-submit]   the submit <button>
 *   #username / #password the two fields (standard ids, also used by the
 *                         browser's autofill).
 */
"use strict";

(function () {
  var form = document.querySelector("[data-login-form]");
  var submit = document.querySelector("[data-login-submit]");
  if (!form || !submit) {
    return; // progressive enhancement: nothing to do.
  }

  form.addEventListener("submit", function (event) {
    var username = form.querySelector("#username");
    var password = form.querySelector("#password");

    // Native validation already runs (novalidate is set, so we do it here):
    // both fields are required, and the password policy lives server-side —
    // we only check that it was actually typed before paying the round trip.
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

    // One submit at a time. If the server is slow the button says so.
    if (submit.dataset.busy === "1") {
      event.preventDefault();
      return;
    }
    submit.dataset.busy = "1";
    submit.disabled = true;
    var original = submit.textContent;
    submit.textContent = "Signing in…";
    // If the submission is somehow blocked (download, offline), restore the
    // button after a generous timeout so the user is never locked out.
    window.setTimeout(function () {
      submit.dataset.busy = "";
      submit.disabled = false;
      submit.textContent = original;
    }, 15000);
  });
})();