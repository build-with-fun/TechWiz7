/* Shared helpers used by every page:
 *   SST.config / SST.vocabulary  the JSON blocks base.html writes into the page
 *   SST.api                      fetch wrapper that reads the error envelope and request id
 *   SST.region                   loading / empty / error states for one panel
 *   SST.toast, SST.error         dismissible notices
 *   SST.el / SST.set / SST.text  element builders (text always goes in via textContent)
 *   SST.lifecycle                cleanup for timers, streams and audio contexts
 * The API is same-origin with an HttpOnly session cookie, so no token lives in JS.
 */
(function (SST) {
  'use strict';

  // Config

  function readJsonBlock(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent || '{}');
    } catch (err) {
      // A broken block is a template bug; log it instead of quietly using defaults.
      if (window.console && console.error) {
        console.error('[SST] could not parse #' + id + ':', err);
      }
      return null;
    }
  }

  var config = readJsonBlock('app-config') || { endpoints: {}, thresholds: {} };
  var vocabulary = readJsonBlock('app-vocabulary') || {};

  function threshold(path, fallback) {
    var node = config.thresholds || {};
    var parts = String(path).split('.');
    for (var i = 0; i < parts.length; i++) {
      if (node === null || typeof node !== 'object' || !(parts[i] in node)) return fallback;
      node = node[parts[i]];
    }
    return node === undefined ? fallback : node;
  }

  // Errors

  function ApiError(message, options) {
    var opts = options || {};
    this.name = 'ApiError';
    this.message = message || 'The request failed.';
    this.code = opts.code || 'unknown_error';
    this.status = opts.status || 0;
    this.details = opts.details === undefined ? null : opts.details;
    this.requestId = opts.requestId || null;
    this.retryable = opts.retryable === undefined ? false : opts.retryable;
    if (Error.captureStackTrace) Error.captureStackTrace(this, ApiError);
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  /* Prefer the server's message, since it knows the actual reason, and add the
     request id so the failure can be found in the log. */
  ApiError.prototype.describe = function () {
    var text = this.message;
    if (this.code && this.code !== 'unknown_error') {
      text += ' (' + this.code + ')';
    }
    if (this.requestId) {
      text += ' Reference: ' + this.requestId + '.';
    }
    return text;
  };

  function envelopeMessage(payload, status) {
    if (payload && typeof payload === 'object' && payload.error && typeof payload.error === 'object') {
      return payload.error;
    }
    return null;
  }

  // API

  var REDIRECT_STATUSES = { 401: true };

  /**
   * SST.api(path, options) -> Promise<data>
   *   options: { method, body, json, headers, timeoutMs, signal, raw }
   * Always rejects with an ApiError, including for network failures, so callers
   * can show a retry message.
   */
  function api(path, options) {
    var opts = options || {};
    var method = (opts.method || 'GET').toUpperCase();
    var headers = Object.assign({ 'Accept': 'application/json' }, opts.headers || {});
    if (method !== 'GET' && method !== 'HEAD') {
      var csrf = document.querySelector('meta[name="csrf-token"]');
      if (csrf) { headers['X-CSRF-Token'] = csrf.content; }
    }
    var init = { method: method, credentials: 'same-origin', headers: headers };

    if (opts.signal) init.signal = opts.signal;

    if (opts.json !== undefined && opts.json !== null) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.json);
    } else if (opts.body !== undefined && opts.body !== null) {
      init.body = opts.body;
    }

    var controller = null;
    var timeoutId = null;
    if (opts.timeoutMs && typeof AbortController !== 'undefined') {
      controller = new AbortController();
      init.signal = controller.signal;
      timeoutId = setTimeout(function () { controller.abort(); }, opts.timeoutMs);
    }

    function clear() { if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; } }

    return fetch(path, init).then(function (response) {
      clear();
      var requestId = response.headers.get('X-Request-Id');

      if (REDIRECT_STATUSES[response.status]) {
        // Session expired: go to the login page once, remembering where we were.
        var here = window.location.pathname + window.location.search;
        if (window.location.pathname !== config.endpoints.login) {
          window.location.assign(config.endpoints.login + '?next=' + encodeURIComponent(here));
        }
        throw new ApiError('Your session has expired. Sign in again to continue.', {
          code: 'unauthorized', status: response.status, requestId: requestId
        });
      }

      var isJson = (response.headers.get('Content-Type') || '').indexOf('json') !== -1;
      if (opts.raw) {
        if (!response.ok) {
          throw new ApiError('The server returned ' + response.status + '.', {
            code: 'http_' + response.status, status: response.status, requestId: requestId
          });
        }
        return response;
      }

      return (isJson ? response.json().catch(function () { return null; }) : response.text().then(function (text) {
        return text ? { message: text } : null;
      })).then(function (payload) {
        if (response.ok) {
          if (payload === null) {
            throw new ApiError('The server sent an empty response.', {
              code: 'empty_response', status: response.status, requestId: requestId
            });
          }
          return payload;
        }
        var error = envelopeMessage(payload, response.status);
        var retryable = response.status === 408 || response.status === 429 || response.status >= 500 ||
                        response.status === 202;
        throw new ApiError(
          (error && error.message) || ('The request failed with status ' + response.status + '.'),
          {
            code: (error && error.code) || ('http_' + response.status),
            status: response.status,
            details: error && error.details,
            requestId: (error && error.request_id) || requestId,
            retryable: retryable
          }
        );
      });
    }, function (err) {
      clear();
      if (err && err.name === 'AbortError') {
        throw new ApiError('The request was cancelled.', { code: 'aborted', retryable: true });
      }
      throw new ApiError(
        'Could not reach the server. Check that the service is running, then try again.',
        { code: 'network_error', retryable: true }
      );
    });
  }

  // Element helpers

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'html') node.innerHTML = value;           // literal markup only, never user data
        else if (key === 'dataset') Object.assign(node.dataset, value);
        else if (key.indexOf('on') === 0 && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else if (value === true) node.setAttribute(key, '');
        else node.setAttribute(key, value);
      });
    }
    append(node, children);
    return node;
  }

  function append(parent, children) {
    if (children === null || children === undefined) return parent;
    if (Array.isArray(children)) {
      children.forEach(function (child) { append(parent, child); });
      return parent;
    }
    parent.appendChild(children instanceof Node ? children : document.createTextNode(String(children)));
    return parent;
  }

  function text(node, value) {
    node.textContent = value === null || value === undefined ? '' : String(value);
    return node;
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function byId(id) { return document.getElementById(id); }

  function qsa(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  /* Glyphs and slugs come from _presentation.html, same as the server-rendered chips. */
  function presentation(kind, value) {
    var table = (vocabulary.presentation || {})[kind] || {};
    return table[value] || { glyph: '\u00b7', slug: 'unknown' };
  }

  // Regions

  /**
   * Gives one panel its own loading, empty and error states.
   *   var region = SST.region(document.querySelector('#results'));
   *   region.loading();
   *   region.error({ message: '…', requestId: '…', retry: fn });
   *   region.render(nodeOrFragment);
   */
  function region(root) {
    if (!root) return null;
    var api = {
      root: root,

      loading: function (label, rows) {
        root.setAttribute('aria-busy', 'true');
        root.dataset.state = 'loading';
        clear(root);
        var wrap = el('div', { class: 'stack stack--tight' });
        wrap.appendChild(el('span', { class: 'visually-hidden', role: 'status', text: label || 'Loading' }));
        for (var i = 0; i < (rows || 4); i++) wrap.appendChild(el('span', { class: 'skeleton skeleton--row' }));
        root.appendChild(wrap);
        return api;
      },

      empty: function (title, body, action) {
        root.setAttribute('aria-busy', 'false');
        root.dataset.state = 'empty';
        clear(root);
        var box = el('div', { class: 'empty', role: 'status' });
        box.appendChild(el('svg', {
          class: 'empty__art', viewBox: '0 0 96 96', 'aria-hidden': 'true', focusable: 'false',
          html: '<rect x="6" y="6" width="84" height="84" rx="16" fill="none" stroke="currentColor" stroke-width="2" opacity="0.35"/>' +
                '<path d="M14 52h8l6-18 8 34 9-46 9 58 8-40 7 22h11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>'
        }));
        box.appendChild(el('p', { class: 'empty__title', text: title || 'Nothing here yet' }));
        if (body) box.appendChild(el('p', { class: 'empty__body', text: body }));
        if (action && action.text) {
          box.appendChild(action.href
            ? el('a', { class: 'btn btn--primary', href: action.href, text: action.text })
            : el('button', { type: 'button', class: 'btn btn--primary', text: action.text, onclick: action.onClick }));
        }
        root.appendChild(box);
        return api;
      },

      error: function (options) {
        var opts = options || {};
        root.setAttribute('aria-busy', 'false');
        root.dataset.state = 'error';
        clear(root);
        var banner = el('div', { class: 'banner banner--error', role: 'alert' });
        banner.appendChild(el('div', { class: 'banner__body' }, [
          el('p', { class: 'banner__title', text: opts.title || 'This panel could not load' }),
          el('p', { class: 'banner__text', text: opts.message || 'The request failed.' }),
          opts.requestId ? el('p', { class: 'banner__text' }, [
            'Reference: ', el('span', { class: 'mono', text: opts.requestId }),
            '. Quote it and we can find the request in the log.'
          ]) : null
        ]));
        if (opts.retry) {
          banner.appendChild(el('div', { class: 'banner__actions' }, [
            el('button', { type: 'button', class: 'btn btn--sm', text: 'Try again', onclick: opts.retry })
          ]));
        }
        root.appendChild(banner);
        return api;
      },

      render: function (content) {
        root.setAttribute('aria-busy', 'false');
        delete root.dataset.state;
        clear(root);
        append(root, content);
        initBars(root);
        return api;
      },

      busy: function (on) {
        root.setAttribute('aria-busy', on === false ? 'false' : 'true');
        return api;
      }
    };
    return api;
  }

  // Notices

  var MAX_TOASTS = 4;

  function toast(kind, message, options) {
    var opts = options || {};
    var stack = byId('toast-stack');
    if (!stack) return null;
    var tone = kind === 'danger' ? 'error' : (kind || 'info');
    var node = el('div', { class: 'toast toast--' + tone, role: tone === 'error' ? 'alert' : 'status' });
    node.appendChild(el('div', { class: 'toast__text', text: message }));
    var close = el('button', {
      type: 'button', class: 'btn btn--ghost btn--icon btn--sm',
      'aria-label': 'Dismiss notification', title: 'Dismiss'
    });
    close.appendChild(el('span', { 'aria-hidden': 'true', text: '\u00d7' }));
    close.addEventListener('click', function () { remove(); });
    node.appendChild(close);
    stack.appendChild(node);

    // Keep the stack short so it does not cover the page; drop the oldest.
    while (stack.children.length > MAX_TOASTS) stack.removeChild(stack.firstChild);

    var timer = null;
    function remove() {
      if (timer) { clearTimeout(timer); timer = null; }
      if (node.parentNode) node.parentNode.removeChild(node);
    }
    if (opts.sticky !== true) timer = setTimeout(remove, opts.ttl || (tone === 'error' ? 12000 : 6000));
    return remove;
  }

  /* Page-level error strip, for failures that affect the whole screen. */
  function showError(message, requestId, retry) {
    var host = byId('page-error');
    if (!host) return;
    clear(host);
    host.hidden = false;
    host.appendChild(el('div', { class: 'banner banner--error', role: 'alert' }, [
      el('div', { class: 'banner__body' }, [
        el('p', { class: 'banner__title', text: 'Something went wrong' }),
        el('p', { class: 'banner__text', text: message }),
        requestId ? el('p', { class: 'banner__text' }, [
          'Reference: ', el('span', { class: 'mono', text: requestId })
        ]) : null
      ]),
      el('div', { class: 'banner__actions' }, [
        retry ? el('button', { type: 'button', class: 'btn btn--sm', text: 'Try again', onclick: retry }) : null,
        el('button', {
          type: 'button', class: 'btn btn--sm btn--ghost', text: 'Dismiss',
          onclick: function () { host.hidden = true; clear(host); }
        })
      ])
    ]));
    host.scrollIntoView({ block: 'nearest' });
  }

  function clearError() {
    var host = byId('page-error');
    if (!host) return;
    host.hidden = true;
    clear(host);
  }

  /* Lifecycle: long-lived resources register here so they are stopped when the
   page is hidden or unloaded. */

  var disposers = [];
  var visibilityHandlers = [];
  var hiddenHandlers = [];
  var armed = false;

  function onTeardown(fn) {
    if (typeof fn !== 'function') return function () {};
    disposers.push(fn);
    arm();
    return function () {
      var at = disposers.indexOf(fn);
      if (at !== -1) disposers.splice(at, 1);
    };
  }

  function runTeardown() {
    var current = disposers.slice();
    disposers.length = 0;
    current.reverse().forEach(function (fn) {
      try { fn(); } catch (err) {
        if (window.console && console.warn) console.warn('[SST] teardown step failed:', err);
      }
    });
  }

  /** Fires when the tab becomes visible again. Used to resume the live loop. */
  function onVisible(fn) { visibilityHandlers.push(fn); arm(); }

  /** Fires when the tab is hidden. The live loop stops capturing here. */
  function onHidden(fn) { hiddenHandlers.push(fn); arm(); }

  function arm() {
    if (armed) return;
    armed = true;

    window.addEventListener('pagehide', runTeardown);
    window.addEventListener('beforeunload', runTeardown);

    document.addEventListener('visibilitychange', function () {
      var handlers = document.hidden ? hiddenHandlers : visibilityHandlers;
      handlers.slice().forEach(function (fn) {
        try { fn(); } catch (err) {
          if (window.console && console.warn) console.warn('[SST] lifecycle handler failed:', err);
        }
      });
    });

    // After a back/forward cache restore the timers are dead, so tell the
    // features to start again.
    window.addEventListener('pageshow', function (event) {
      if (event.persisted) {
        dispatch('pageshow-restored');
      }
    });
  }

  var named = {};
  function on(name, fn) { (named[name] = named[name] || []).push(fn); }
  function dispatch(name, payload) {
    (named[name] || []).slice().forEach(function (fn) {
      try { fn(payload); } catch (err) {
        if (window.console && console.warn) console.warn('[SST] handler ' + name + ' failed:', err);
      }
    });
  }

  // Theme

  var THEMES = ['dark', 'light'];

  function currentTheme() {
    var theme = document.documentElement.getAttribute('data-theme');
    return THEMES.indexOf(theme) >= 0 ? theme : 'light';
  }

  function setTheme(theme) {
    var next = THEMES.indexOf(theme) >= 0 ? theme : 'light';
    document.documentElement.setAttribute('data-theme', next);
    // The server reads the same cookie, avoiding a flash of the default theme.
    document.cookie = 'sst_theme_v2=' + next + '; path=/; max-age=31536000; SameSite=Lax' +
      (window.location.protocol === 'https:' ? '; Secure' : '');
    qsa('[data-theme-toggle]').forEach(function (button) {
      button.setAttribute('aria-pressed', next === 'light' ? 'true' : 'false');
    });
    dispatch('themechange', next);
  }

  function initThemeToggle() {
    qsa('[data-theme-toggle]').forEach(function (button) {
      button.addEventListener('click', function () {
        setTheme(currentTheme() === 'light' ? 'dark' : 'light');
      });
    });
  }

  /* Bars carry their width in data-bar-width because the CSP blocks style=""
     attributes. Setting it through the CSSOM is allowed. */
  function initBars(root) {
    qsa('[data-bar-width]', root || document).forEach(function (fill) {
      var pct = parseFloat(fill.getAttribute('data-bar-width'));
      fill.style.width = (isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0) + '%';
      fill.removeAttribute('data-bar-width');
    });
  }

  // Navigation

  /* Below 1025px the sidebar is a drawer. Escape, the scrim and the close button
     shut it and return focus to the toggle. */
  function initNav() {
    var app = document.querySelector('.app');
    var nav = byId('app-nav');
    var toggle = document.querySelector('[data-nav-toggle]');
    var scrim = document.querySelector('[data-nav-scrim]');
    if (!app || !nav || !toggle) return;

    function isOpen() { return app.getAttribute('data-nav-open') === 'true'; }
    function setOpen(open) {
      app.setAttribute('data-nav-open', open ? 'true' : 'false');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (scrim) scrim.hidden = !open;
      document.documentElement.classList.toggle('is-nav-open', open);
      if (open) {
        var first = nav.querySelector('a[href]');
        if (first) first.focus();
      } else if (document.activeElement && nav.contains(document.activeElement)) {
        toggle.focus();
      }
    }

    toggle.addEventListener('click', function () { setOpen(!isOpen()); });
    qsa('[data-nav-close]').forEach(function (button) {
      button.addEventListener('click', function () { setOpen(false); });
    });
    if (scrim) scrim.addEventListener('click', function () { setOpen(false); });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && isOpen()) setOpen(false);
    });
    var wide = window.matchMedia('(min-width: 1025px)');
    var onWide = function (event) { if (event.matches && isOpen()) setOpen(false); };
    if (wide.addEventListener) wide.addEventListener('change', onWide);
  }

  /* On phones tables turn into cards (design.css), so copy each header onto its cells
     as a label. */
  function initTableLabels(root) {
    qsa('table.data', root || document).forEach(function (table) {
      var heads = qsa('thead th', table).map(function (th) {
        return th.textContent.replace(/\s+/g, ' ').trim();
      });
      qsa('tbody tr', table).forEach(function (row) {
        Array.prototype.forEach.call(row.children, function (cell, index) {
          if (!cell.hasAttribute('data-label') && heads[index]) cell.setAttribute('data-label', heads[index]);
        });
      });
    });
  }

  /* Count data-count-up numbers up from zero on load, unless reduced motion is on. */
  function motionAllowed() {
    return !window.matchMedia('(prefers-reduced-motion: reduce)').matches &&
      document.documentElement.getAttribute('data-motion') !== 'reduced';
  }

  function initCountUp(root) {
    if (!motionAllowed() || !window.requestAnimationFrame) return;
    qsa('[data-count-up]', root || document).forEach(function (el) {
      var text = el.textContent.trim();
      var target = parseFloat(text.replace(/[^0-9.]/g, ''));
      if (!isFinite(target) || target === 0) return;
      var decimals = (text.split('.')[1] || '').replace(/[^0-9]/g, '').length;
      var suffix = text.replace(/[0-9.,\s]/g, '');
      var start = null;
      function frame(now) {
        if (start === null) start = now;
        var t = Math.min(1, (now - start) / 1000);
        var eased = 1 - Math.pow(1 - t, 3);
        el.textContent = (target * eased).toLocaleString('en-US', {
          minimumFractionDigits: decimals, maximumFractionDigits: decimals
        }) + suffix;
        if (t < 1) window.requestAnimationFrame(frame);
        else el.textContent = text;
      }
      window.requestAnimationFrame(frame);
    });
  }

  // Delegated handlers

  /* Retry buttons in error blocks and password eye buttons, including ones added later. */
  function initDelegates() {
    document.addEventListener('click', function (event) {
      var reveal = event.target.closest ? event.target.closest('[data-reveal]') : null;
      if (reveal) {
        var field = document.getElementById(reveal.getAttribute('data-reveal'));
        if (field) {
          var show = field.type === 'password';
          field.type = show ? 'text' : 'password';
          reveal.setAttribute('aria-pressed', show ? 'true' : 'false');
          reveal.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
          field.focus();
        }
        return;
      }
      var target = event.target.closest ? event.target.closest('[data-retry]') : null;
      if (target) {
        event.preventDefault();
        var button = target;
        if (button.dataset.retryUrl) window.location.assign(button.dataset.retryUrl);
        else window.location.reload();
      }
    });
    // Hide a shown password again before sending, so the browser saves it as a password.
    document.addEventListener('submit', function (event) {
      var buttons = event.target.querySelectorAll ? event.target.querySelectorAll('[data-reveal][aria-pressed="true"]') : [];
      Array.prototype.forEach.call(buttons, function (button) { button.click(); });
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initDelegates();
    initBars();
    initThemeToggle();
    initNav();
    initTableLabels();
    initCountUp();
  });

  // Public API

  SST.config = config;
  SST.vocabulary = vocabulary;
  SST.threshold = threshold;
  SST.endpoint = function (name) { return (config.endpoints || {})[name] || null; };
  SST.endpointFor = function (name, vars) {
    var url = SST.endpoint(name);
    if (!url) return null;
    Object.keys(vars || {}).forEach(function (key) {
      url = url.replace('__' + key.replace(/([a-z])([A-Z])/g, '$1_$2').toUpperCase() + '__', encodeURIComponent(vars[key]));
    });
    return url;
  };

  SST.api = api;
  SST.ApiError = ApiError;

  SST.el = el;
  SST.append = append;
  SST.text = text;
  SST.clear = clear;
  SST.byId = byId;
  SST.qsa = qsa;
  SST.initBars = initBars;
  SST.initTableLabels = initTableLabels;
  SST.on = on;
  SST.dispatch = dispatch;
  SST.region = region;
  SST.presentation = presentation;

  SST.toast = toast;
  SST.error = { show: showError, clear: clearError };

  SST.lifecycle = {
    onTeardown: onTeardown,
    runTeardown: runTeardown,
    onVisible: onVisible,
    onHidden: onHidden,
    on: on,
    dispatch: dispatch
  };

  SST.theme = { get: currentTheme, set: setTheme };
})(window.SST = window.SST || {});
