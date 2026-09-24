/* ============================================================================
 * static/js/core.js — the small shared layer under every screen.
 *
 * What it provides, and why each piece exists:
 *   SST.config / SST.vocabulary  Read the inert JSON blocks base.html emits.
 *   SST.api                      fetch() that understands the error envelope
 *                                ({error:{code,message,details,request_id}}),
 *                                keeps the X-Request-Id visible to the operator,
 *                                and never lets a rejected promise escape.
 *   SST.region                   The loading / empty / error states for one part
 *                                of the page. This is the thing that stops a
 *                                failed panel from blanking the console.
 *   SST.toast, SST.error         Non-blocking notices, always dismissible.
 *   SST.el / SST.set / SST.text  Element builders. Text goes in through
 *                                textContent: a filename from an upload is user
 *                                input and is never interpolated as markup.
 *   SST.lifecycle                Teardown registry. Anything holding a timer, a
 *                                MediaStream, an AudioContext or a socket
 *                                registers here; pagehide and overnight-tab
 *                                handling are then automatic rather than
 *                                remembered per feature.
 *
 * Network behaviour worth knowing: the API is same-origin, cookie-authenticated
 * (session cookie is HttpOnly + SameSite=Lax per config/auth.json), so every
 * request sends credentials and no token is handled in JavaScript.
 * ========================================================================== */
(function (SST) {
  'use strict';

  /* --------------------------------------------------------------- config - */

  function readJsonBlock(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent || '{}');
    } catch (err) {
      // A malformed block is a build-time bug, not a runtime condition. Say so
      // once, loudly, rather than silently running with defaults.
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

  /* ---------------------------------------------------------------- error - */

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

  /* A sentence an operator can act on. The server's `message` is preferred --
     it knows whether a duplicate is a duplicate or a quality rejection -- and
     the request id is appended because that is what makes a support question
     answerable. */
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

  /* ------------------------------------------------------------------ api - */

  var REDIRECT_STATUSES = { 401: true };

  /**
   * SST.api(path, options) -> Promise<data>
   *   options: { method, body, json, headers, timeoutMs, signal, raw }
   * Rejects with an ApiError. Never rejects with a TypeError from fetch: a
   * dropped connection while the laptop sleeps is a normal event in a console
   * left open overnight, and it must render as a retryable message.
   */
  function api(path, options) {
    var opts = options || {};
    var method = (opts.method || 'GET').toUpperCase();
    var headers = Object.assign({ 'Accept': 'application/json' }, opts.headers || {});
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
        // The session expired while the tab sat open. Send the operator to the
        // login page with the page they were on, once — not in a loop.
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
        'Could not reach the server. Check that the service is running — this message is safe to retry.',
        { code: 'network_error', retryable: true }
      );
    });
  }

  /* ------------------------------------------------------------------- el - */

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

  /* The presentation table (glyphs, slugs) is emitted by _presentation.html so
     a client-rendered chip cannot drift from a server-rendered one. */
  function presentation(kind, value) {
    var table = (vocabulary.presentation || {})[kind] || {};
    return table[value] || { glyph: '\u00b7', slug: 'unknown' };
  }

  /* -------------------------------------------------------------- regions - */

  /**
   * Wrap one part of a page so it can show its own loading, empty and error
   * states without touching its neighbours.
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
            ' — quote this and the exact request can be found in the log.'
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
        return api;
      },

      busy: function (on) {
        root.setAttribute('aria-busy', on === false ? 'false' : 'true');
        return api;
      }
    };
    return api;
  }

  /* --------------------------------------------------------------- notices - */

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

    // A deep stack of notices is noise, and on the live console it would cover
    // the console itself. Oldest goes first.
    while (stack.children.length > MAX_TOASTS) stack.removeChild(stack.firstChild);

    var timer = null;
    function remove() {
      if (timer) { clearTimeout(timer); timer = null; }
      if (node.parentNode) node.parentNode.removeChild(node);
    }
    if (opts.sticky !== true) timer = setTimeout(remove, opts.ttl || (tone === 'error' ? 12000 : 6000));
    return remove;
  }

  /* The page-level error strip. Used when a failure belongs to the whole screen
     rather than one panel, e.g. the search endpoint itself being down. */
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

  /* ------------------------------------------------------------- lifecycle -
   * Every long-lived resource registers here. This is the answer to "the tab
   * was left open overnight": nothing relies on the author remembering to stop
   * its own timer on unload.
   * ---------------------------------------------------------------------- */

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

  /**
   * Fires when the tab is hidden. The live loop uses this to stop capturing and
   * posting: an overnight background tab must not hold the microphone open or
   * keep posting windows nobody is watching.
   */
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

    // A bfcache restore re-arms nothing by itself: the page comes back with its
    // timers already dead. Tell the features so they can re-register rather than
    // sit there looking alive.
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

  /* ---------------------------------------------------------------- theme - */

  var THEME_COOKIE = 'sst_theme';

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function setTheme(theme) {
    var next = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    // Cookie, not localStorage: the server renders data-theme on the first byte,
    // so there is no flash of the wrong theme before scripts run.
    document.cookie = THEME_COOKIE + '=' + next + '; path=/; max-age=31536000; SameSite=Lax';
    qsa('[data-theme-toggle]').forEach(function (button) {
      button.setAttribute('aria-pressed', next === 'light' ? 'true' : 'false');
      button.title = next === 'light' ? 'Switch to the dark console' : 'Switch to the light theme';
      var slot = button.querySelector('[data-theme-icon]');
      if (slot) {
        slot.innerHTML = '';
        var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('width', '16'); svg.setAttribute('height', '16');
        svg.setAttribute('fill', 'none'); svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '1.8'); svg.setAttribute('stroke-linecap', 'round');
        svg.setAttribute('stroke-linejoin', 'round'); svg.setAttribute('aria-hidden', 'true');
        var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', next === 'light'
          ? 'M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z'
          : 'M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4');
        svg.appendChild(path);
        if (next === 'light') {
          var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('cx', '12'); circle.setAttribute('cy', '12'); circle.setAttribute('r', '4');
          svg.appendChild(circle);
          svg.removeAttribute('fill');
        } else {
          svg.setAttribute('fill', 'none');
        }
        slot.appendChild(svg);
      }
    });
  }

  function initTheme() {
    qsa('[data-theme-toggle]').forEach(function (button) {
      button.addEventListener('click', function () {
        setTheme(currentTheme() === 'light' ? 'dark' : 'light');
      });
    });
  }

  /* ------------------------------------------------------------ delegated - */

  /* Retry buttons that the server-rendered error blocks emit. Delegated so a
     block inserted later still works. */
  function initDelegates() {
    document.addEventListener('click', function (event) {
      var target = event.target.closest ? event.target.closest('[data-retry]') : null;
      if (target) {
        event.preventDefault();
        var button = target;
        if (button.dataset.retryUrl) window.location.assign(button.dataset.retryUrl);
        else window.location.reload();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initDelegates();
  });

  /* ------------------------------------------------------------------ API - */

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
