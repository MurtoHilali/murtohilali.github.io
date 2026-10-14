/*
 * Progressive extras for /lab/. The rows, details, specs and links are all
 * in the HTML (built from _data/projects.yml), so the page works without
 * this file. This adds:
 *   - opening a row from the URL hash (/lab/#pac-maps)
 *   - the status-block clock and click-to-cycle readout
 *   - the ASCII monitor in an open row
 *   - sound, off by default
 * Motion stops under prefers-reduced-motion, and the animation loop pauses
 * while the tab is hidden.
 */
(function () {
  'use strict';

  document.documentElement.classList.add('lab-js');

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var canHover = window.matchMedia('(hover: hover)').matches;
  var rows = Array.prototype.slice.call(document.querySelectorAll('.lab-row'));

  /* ------------------------------------------------------------- sound */

  var SOUND_KEY = 'lab-sound';
  var soundOn = false;
  var audioCtx = null;
  try { soundOn = window.localStorage.getItem(SOUND_KEY) === 'on'; } catch (e) {}

  // Only ever called from a user gesture, so the context starts running.
  function ensureAudio() {
    if (!audioCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      audioCtx = new AC();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
  }

  function playSound(type, index) {
    if (!soundOn || !audioCtx || audioCtx.state !== 'running') return;
    try {
      var now = audioCtx.currentTime;
      var osc = audioCtx.createOscillator();
      var gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (type === 'hover') {
        // Soft mechanical tap; pitch steps down by row
        var base = 500 - ((index || 0) * 80);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(base * 1.5, now);
        osc.frequency.exponentialRampToValueAtTime(base, now + 0.015);
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.04, now + 0.002);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.02);
        osc.start();
        osc.stop(now + 0.025);
      } else {
        // Deeper, chunky clack
        osc.type = 'square';
        osc.frequency.setValueAtTime(150, now);
        osc.frequency.exponentialRampToValueAtTime(80, now + 0.05);
        gain.gain.setValueAtTime(0.03, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
        osc.start();
        osc.stop(now + 0.05);
      }
    } catch (e) {}
  }

  var soundBtn = document.getElementById('sound-toggle');
  function renderSoundButton() {
    if (!soundBtn) return;
    soundBtn.setAttribute('aria-pressed', soundOn ? 'true' : 'false');
    var state = soundBtn.querySelector('[data-sound-state]');
    if (state) state.textContent = soundOn ? 'on' : 'off';
  }
  if (soundBtn) {
    soundBtn.hidden = false;
    renderSoundButton();
    soundBtn.addEventListener('click', function () {
      soundOn = !soundOn;
      try { window.localStorage.setItem(SOUND_KEY, soundOn ? 'on' : 'off'); } catch (e) {}
      if (soundOn) { ensureAudio(); playSound('click'); }
      renderSoundButton();
    });
  }

  // Sound remembered as on: wait for the visitor's first gesture before
  // creating the AudioContext.
  if (soundOn) {
    var unlock = function () {
      ensureAudio();
      window.removeEventListener('pointerdown', unlock, true);
      window.removeEventListener('keydown', unlock, true);
    };
    window.addEventListener('pointerdown', unlock, true);
    window.addEventListener('keydown', unlock, true);
  }

  /* -------------------------------------------------------------- rows */

  rows.forEach(function (row, i) {
    var summary = row.querySelector('summary');
    if (canHover) summary.addEventListener('mouseenter', function () { playSound('hover', i % 6); });
    summary.addEventListener('click', function () { playSound('click'); });
    row.addEventListener('toggle', function () {
      if (row.open) {
        // One row open at a time, for browsers without <details name>
        rows.forEach(function (other) { if (other !== row && other.open) other.open = false; });
      }
      refreshMonitors();
    });
  });

  document.querySelectorAll('.lab-panel .btn').forEach(function (btn) {
    btn.addEventListener('mouseenter', function () { playSound('hover'); });
    btn.addEventListener('click', function () { playSound('click'); });
  });

  function openFromHash() {
    var id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch (e) { return; }
    if (!id) return;
    var row = document.getElementById(id);
    if (!row || !row.classList.contains('lab-row')) return;
    row.open = true;
    row.scrollIntoView({ block: 'start', behavior: reduceMotion.matches ? 'auto' : 'smooth' });
  }
  openFromHash();
  window.addEventListener('hashchange', openFromHash);

  /* ----------------------------------------------------- ASCII monitors */

  var DENSITY = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
  var W = 46;
  var H = 22;

  function monitorFrame(type, time) {
    var out = '';
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var val = 0;
        var nx = (x / W) * 2 - 1;
        var ny = (y / H) * 2 - 1;

        if (type === 'cnn') {
          var isGridLine = Math.abs(Math.sin(nx * 8)) < 0.2 || Math.abs(Math.sin(ny * 8)) < 0.2;
          var dx = Math.abs(nx - Math.sin(time * 1.5) * 0.6);
          var dy = Math.abs(ny - Math.cos(time * 1.1) * 0.6);
          var inFilter = dx < 0.25 && dy < 0.25;
          var isFilterBorder = dx < 0.3 && dy < 0.3 && !inFilter;
          if (isFilterBorder) val = 0.9;
          else if (inFilter) val = 0.6 + Math.sin(nx * 30 + ny * 30 - time * 20) * 0.4;
          else if (isGridLine) val = 0.15;
          else val = 0.02;
        } else if (type === 'neural') {
          var cx1 = nx + Math.sin(time * 0.8) * 0.5;
          var cy1 = ny + Math.cos(time * 1.2) * 0.5;
          var cx2 = nx - Math.cos(time * 1.5) * 0.3;
          var cy2 = ny - Math.sin(time * 0.7) * 0.3;
          var d1 = Math.sqrt(cx1 * cx1 + cy1 * cy1);
          var d2 = Math.sqrt(cx2 * cx2 + cy2 * cy2);
          val = (Math.sin(d1 * 8 - time * 3) + Math.cos(d2 * 10 + time * 2) + 2) / 4;
        } else if (type === 'dna') {
          var wave1 = Math.sin(ny * 4 + time * 3) * 0.6;
          var wave2 = Math.sin(ny * 4 + time * 3 + Math.PI) * 0.6;
          if (Math.abs(nx - wave1) < 0.1 || Math.abs(nx - wave2) < 0.1) val = 0.9;
          else if (Math.abs(wave1 - wave2) > 0.2 && nx > Math.min(wave1, wave2) && nx < Math.max(wave1, wave2) && y % 2 === 0) val = 0.4;
          else val = Math.max(0, 0.1 - Math.abs(nx));
        } else if (type === 'scan') {
          var angle = Math.atan2(ny, nx);
          var dist = Math.sqrt(nx * nx + ny * ny);
          var radar = (angle + time * 2) % (Math.PI * 2);
          if (radar < 0) radar += Math.PI * 2;
          var ring = Math.abs(Math.sin(dist * 15 - time * 4));
          val = (1 - radar / (Math.PI * 2)) * ring + (dist < 0.8 && dist > 0.75 ? 0.8 : 0);
        }

        val = Math.max(0, Math.min(1, val));
        out += DENSITY[Math.floor(val * (DENSITY.length - 1))];
      }
      out += '\n';
    }
    return out;
  }

  var monitors = rows.map(function (row) {
    var pre = row.querySelector('.ascii-animator');
    return pre ? { row: row, pre: pre, type: pre.getAttribute('data-type') } : null;
  }).filter(Boolean);

  /* ------------------------------------------------- animation loop */

  var time = 0;
  var rafId = null;

  function drawMonitors() {
    monitors.forEach(function (m) {
      if (m.row.open) m.pre.textContent = monitorFrame(m.type, time);
    });
  }
  function anyMonitorOpen() {
    return monitors.some(function (m) { return m.row.open; });
  }
  function shouldAnimate() {
    return !reduceMotion.matches && document.visibilityState === 'visible' && anyMonitorOpen();
  }

  function tick() {
    rafId = null;
    time += 0.015;
    drawMonitors();
    if (shouldAnimate()) rafId = window.requestAnimationFrame(tick);
  }
  function start() {
    if (rafId === null && shouldAnimate()) rafId = window.requestAnimationFrame(tick);
  }
  function stop() {
    if (rafId !== null) { window.cancelAnimationFrame(rafId); rafId = null; }
  }

  // Called when a row opens or closes. With reduced motion, an opened
  // monitor gets one static frame instead of the loop.
  function refreshMonitors() {
    if (reduceMotion.matches) { drawMonitors(); return; }
    start();
  }

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') start(); else stop();
  });

  function onMotionPreference() {
    if (reduceMotion.matches) {
      stop();
      drawMonitors();
    } else {
      start();
    }
  }
  if (reduceMotion.addEventListener) reduceMotion.addEventListener('change', onMotionPreference);

  drawMonitors();
  start();

  /* ---------------------------------------------------- status block */

  var statusBlock = document.getElementById('status-block');
  var statusReadout = document.getElementById('status-readout');
  var STATUS_GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/:.,+*=#';
  var SESSION_START = Date.now();
  var SCRAMBLE_MS = 420;
  function pad2(n) { return String(n).padStart(2, '0'); }

  var STATUS_PAGES = [
    function () {
      return [
        'LOC: TORONTO, CA',
        'SYS: PROTEINLOOP',
        'TIME: ' + new Date().toLocaleTimeString('en-GB', { timeZone: 'America/Toronto', hour12: false })
      ];
    },
    function () {
      var secs = Math.floor((Date.now() - SESSION_START) / 1000);
      return [
        'IDX: ' + rows.length + ' PROJECTS',
        'RUN: ' + document.querySelectorAll('.lab-status-active').length + ' ACTIVE',
        'SESS: ' + pad2(Math.floor(secs / 60)) + ':' + pad2(secs % 60)
      ];
    },
    function () {
      return ['ORG: UWATERLOO', 'DOM: COMPBIO / ML', 'NET: MURTO.CO'];
    }
  ];

  var statusPage = 0;
  var scrambleStart = 0;

  // Characters resolve left to right, so each line reads like it's decoding.
  function scrambleLine(line, prog) {
    var out = '';
    for (var i = 0; i < line.length; i++) {
      var ch = line[i];
      if (ch === ' ' || prog >= 0.25 + (i / line.length) * 0.75) out += ch;
      else out += STATUS_GLYPHS[Math.floor(Math.random() * STATUS_GLYPHS.length)];
    }
    return out;
  }

  function drawStatus() {
    if (!statusReadout) return;
    var elapsed = performance.now() - scrambleStart;
    var scrambling = !reduceMotion.matches && elapsed < SCRAMBLE_MS;
    var lines = STATUS_PAGES[statusPage]().map(function (l) {
      return scrambling ? scrambleLine(l, elapsed / SCRAMBLE_MS) : l;
    });
    // Block cursor on the last line; it blinks unless motion is reduced
    var cursorOn = reduceMotion.matches || Math.floor(Date.now() / 500) % 2 === 0;
    lines[lines.length - 1] += cursorOn ? '█' : ' ';
    statusReadout.textContent = lines.join('\n');
    if (scrambling) window.requestAnimationFrame(drawStatus);
  }

  if (statusBlock) {
    statusBlock.addEventListener('click', function () {
      playSound('click');
      statusPage = (statusPage + 1) % STATUS_PAGES.length;
      scrambleStart = performance.now();
      drawStatus();
    });
    if (canHover) statusBlock.addEventListener('mouseenter', function () { playSound('hover'); });
  }

  scrambleStart = performance.now();
  drawStatus();
  // Keeps the clock (and the cursor blink) current between scrambles
  window.setInterval(function () {
    if (document.visibilityState === 'visible' && performance.now() - scrambleStart >= SCRAMBLE_MS) drawStatus();
  }, 250);
})();
