/* Value formatting shared by all page scripts, so the live console and the event
 * page always show numbers the same way. Pure functions on window.SST. */
(function (SST) {
  'use strict';

  var NUM = typeof Intl !== 'undefined' ? Intl.NumberFormat(undefined) : null;

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  /* Always three decimals so live values do not jump around. */
  function confidence(v, places) {
    if (!isNum(v)) return '--';
    return v.toFixed(places === undefined ? 3 : places);
  }

  function percent(v, places) {
    if (!isNum(v)) return '--';
    return (v * 100).toFixed(places === undefined ? 1 : places) + '%';
  }

  function number(v, places) {
    if (!isNum(v)) return '--';
    if (places === undefined) return NUM ? NUM.format(v) : String(v);
    return v.toFixed(places);
  }

  function bytes(n) {
    if (!isNum(n)) return '--';
    if (n < 1024) return n + ' B';
    if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
    return (n / 1073741824).toFixed(2) + ' GB';
  }

  function ms(v) {
    if (!isNum(v)) return '--';
    return Math.round(v) + ' ms';
  }

  /* m:ss.s, as used by the player and the live console. */
  function duration(seconds) {
    if (!isNum(seconds)) return '--';
    var sign = seconds < 0 ? '-' : '';
    var s = Math.abs(seconds);
    var m = Math.floor(s / 60);
    var rest = s - m * 60;
    return sign + m + ':' + (rest < 10 ? '0' : '') + rest.toFixed(1);
  }

  /* The API sends ISO-8601 UTC. Show it in the local timezone and name the zone. */
  function timestamp(iso, opts) {
    if (!iso) return '--';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    var o = opts || {};
    if (o.dateOnly) {
      return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: '2-digit' });
    }
    return d.toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  }

  function timeOnly(iso) {
    if (!iso) return '--';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  /* "4 minutes ago"; the exact time goes in the title attribute. */
  function relative(iso) {
    if (!iso) return '--';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    var secs = (Date.now() - d.getTime()) / 1000;
    var future = secs < 0;
    secs = Math.abs(secs);
    var text;
    if (secs < 45) text = 'moments';
    else if (secs < 90) text = '1 minute';
    else if (secs < 3600) text = Math.round(secs / 60) + ' minutes';
    else if (secs < 5400) text = '1 hour';
    else if (secs < 86400) text = Math.round(secs / 3600) + ' hours';
    else if (secs < 129600) text = '1 day';
    else text = Math.round(secs / 86400) + ' days';
    return future ? 'in ' + text : text + ' ago';
  }

  function dateInput(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    var m = String(d.getMonth() + 1).padStart(2, '0');
    var day = String(d.getDate()).padStart(2, '0');
    return d.getFullYear() + '-' + m + '-' + day;
  }

  /* Shortens a filename for a table cell but keeps the extension. */
  function truncateFilename(name, max) {
    if (!name) return '';
    var limit = max || 42;
    if (name.length <= limit) return name;
    var dot = name.lastIndexOf('.');
    var ext = dot > 0 ? name.slice(dot) : '';
    var stem = dot > 0 ? name.slice(0, dot) : name;
    return stem.slice(0, Math.max(4, limit - ext.length - 1)) + '\u2026' + ext;
  }

  /* Signed Python minus Teachable Machine difference. The thresholds use the
     magnitude; the sign shows which model was more confident. */
  function signed(v, places) {
    if (!isNum(v)) return '--';
    var text = v.toFixed(places === undefined ? 3 : places);
    return v > 0 ? '+' + text : text;
  }

  function titleCase(slug) {
    if (!slug) return '';
    return String(slug).replace(/[_-]+/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function plural(count, singular, pluralForm) {
    return count === 1 ? singular : (pluralForm || singular + 's');
  }

  SST.format = {
    confidence: confidence,
    percent: percent,
    number: number,
    bytes: bytes,
    ms: ms,
    duration: duration,
    timestamp: timestamp,
    timeOnly: timeOnly,
    relative: relative,
    dateInput: dateInput,
    truncateFilename: truncateFilename,
    signed: signed,
    titleCase: titleCase,
    plural: plural,
    isNum: isNum
  };
})(window.SST = window.SST || {});
