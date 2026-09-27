/* Public page motion: scroll reveals, count-ups and the navigation's scrolled state.
   Everything is progressive: without this file, or with reduced motion, the page is
   complete and static. */
(function () {
  'use strict';

  var root = document.documentElement;
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function countUp(el) {
    var target = parseFloat(String(el.getAttribute('data-count-to') || el.textContent).replace(/[^0-9.]/g, ''));
    if (!isFinite(target)) return;
    var suffix = el.getAttribute('data-count-suffix') || '';
    var decimals = parseInt(el.getAttribute('data-count-decimals') || '0', 10);
    var start = null;
    var duration = 1100;
    function frame(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (target * eased).toLocaleString('en-US', {
        minimumFractionDigits: decimals, maximumFractionDigits: decimals
      }) + suffix;
      if (t < 1) window.requestAnimationFrame(frame);
    }
    window.requestAnimationFrame(frame);
  }

  function reveal(el) {
    el.classList.add('is-visible');
    Array.prototype.forEach.call(el.querySelectorAll('[data-count-to]'), countUp);
  }

  if (!reduced && 'IntersectionObserver' in window) {
    root.classList.add('lp-motion');
    /* Children of a [data-reveal-stagger] block come in one after another. */
    Array.prototype.forEach.call(document.querySelectorAll('[data-reveal-stagger]'), function (group) {
      Array.prototype.forEach.call(group.children, function (child, index) {
        child.setAttribute('data-reveal', '');
        child.style.setProperty('--reveal-delay', (index * 90) + 'ms');
      });
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          reveal(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.12 });
    Array.prototype.forEach.call(document.querySelectorAll('[data-reveal]'), function (el) {
      observer.observe(el);
    });
  }

  /* The pill navigation turns solid once the hero has scrolled away. */
  var nav = document.querySelector('.lp-nav-wrap');
  var hero = document.querySelector('.lp-hero');
  if (nav && hero && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      nav.classList.toggle('is-scrolled', !entries[0].isIntersecting);
    }, { rootMargin: '-80px 0px 0px 0px' }).observe(hero);
  }
})();
