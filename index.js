// xG Pitch Interactive Canvas with suspenseful animations
(function() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const promptEl = document.getElementById('prompt');
  const scrollCue = document.getElementById('scroll-cue');
  const citationEl = document.getElementById('citation');

  const DPR = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  // StatsBomb logical field dimensions in arbitrary units (length x width)
  const SB_LEN = 120; // x in [0,120], opponent goal at x=120
  const SB_WID = 80;  // y in [0,80], center line at y=40
  // Visual-only depth stretch (does not change coordinate space)
  const DEPTH_STRETCH = 1.2;

  // xG model predictor (loaded asynchronously from model/xg_mlp_web.json)
  let xgPredictor = null;
  (function loadModel(){
    const url = 'model/xg_mlp_web.json';
    if (typeof loadXgModel === 'function') {
      loadXgModel(url).then(p => { xgPredictor = p; }).catch(err => {
        console.error('Failed to load xG model:', err);
      });
    } else {
      console.warn('xG runtime not loaded yet. Ensure model/xg_mlp_infer.js is included before index.js');
    }
  })();

  let state = {
    initialized: false, // when true, the pitch is shown
    width: 0,
    height: 0,
    pitchRect: { x: 0, y: 0, w: 0, h: 0 },
    landscape: true,
    // Visible SB-world window near opponent goal
    sbView: { minX: 96, maxX: 120, minY: 0, maxY: 80 },
    points: [], // { u: 0..1 relative to pitchRect axes, v: 0..1, xg: number }
    seq: null,  // active click sequence animation or null
    typingPromptDone: false,
    firstSequenceCompleted: false
  };

  // Typewriter prompt setup
  function startPromptTyping() {
    if (!promptEl) return;
    const fullText = 'click anywhere to begin';
    promptEl.classList.add('typing');
    promptEl.innerHTML = '<span class="text"></span><span class="caret"></span>';
    const textSpan = promptEl.querySelector('.text');
    const caret = promptEl.querySelector('.caret');
    let i = 0;
    const step = () => {
      if (i <= fullText.length) {
        textSpan.textContent = fullText.slice(0, i);
        i++;
        setTimeout(step, i < 6 ? 140 : i < 16 ? 75 : 55);
      } else {
        // done typing
        promptEl.classList.remove('typing');
        if (caret) caret.remove();
        promptEl.classList.add('soft-pulse');
        state.typingPromptDone = true;
      }
    };
    step();
  }

  function onResize() {
    const w = Math.ceil(window.innerWidth);
    const h = Math.ceil(window.innerHeight);

    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.floor(w * DPR);
    canvas.height = Math.floor(h * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

    state.width = w;
    state.height = h;

    // Decide orientation by viewport orientation
    state.landscape = w >= h; // if vertical screen, false

    // Determine target aspect by orientation for the selected SB view window
    const view = state.sbView;
    const viewLen = Math.max(1e-6, view.maxX - view.minX); // along SB x
    const viewWid = Math.max(1e-6, view.maxY - view.minY); // along SB y
    const aspect = state.landscape ? (viewWid / viewLen) : (viewLen / viewWid);

    // Fit pitch into viewport maintaining aspect, with small margin
    const margin = 0;
    let availW = w - margin * 2;
    let availH = h - margin * 2;

    let pitchW, pitchH;
    if (state.landscape) {
      // Landscape: goal line must be at the top, fill vertical depth fully.
      pitchH = availH;                 // fill depth axis
      pitchW = pitchH * aspect;        // preserve aspect
      // Align goal line to the top edge -> y = 0
      state.pitchRect = {
        x: (w - pitchW) / 2,           // center horizontally to show as much lateral as available
        y: 0,
        w: pitchW,
        h: pitchH
      };
    } else {
      // Portrait (phone): goal line must be on the right, fill horizontal depth fully.
      pitchW = availW;                 // fill depth axis
      pitchH = pitchW / aspect;        // preserve aspect
      // Align goal line to the right edge -> x = w - pitchW
      state.pitchRect = {
        x: w - pitchW,
        y: (h - pitchH) / 2,           // center vertically to show as much as available
        w: pitchW,
        h: pitchH
      };
    }

    draw();
  }

  function clear() {
    ctx.clearRect(0, 0, state.width, state.height);
    // dark background with subtle vignette
    const g = ctx.createRadialGradient(
      state.width/2, state.height/2, Math.min(state.width, state.height)*0.1,
      state.width/2, state.height/2, Math.max(state.width, state.height)*0.8
    );
    g.addColorStop(0, '#0b0f14');
    g.addColorStop(1, '#06090d');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, state.width, state.height);
  }

  function strokeLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // --- SB mapping helpers ---
  function sbToCanvas(xSB, ySB) {
    const { x, y, w, h } = state.pitchRect;
    const v = state.sbView;
    const nx = (xSB - v.minX) / (v.maxX - v.minX); // 0..1 along SB x
    const ny = (ySB - v.minY) / (v.maxY - v.minY); // 0..1 along SB y
    if (state.landscape) {
      // Opponent goal line at top, (120,0) top-left, (120,80) top-right
      // nx=1 maps to top edge; y increases left->right across the top
      const px = x + ny * w;           // left->right from y
      const py = y + (1 - nx) * h;     // top at nx=1, bottom at nx=0
      return [px, py];
    } else {
      // Phone/portrait: opponent goal on the right, (120,0) top-right, (120,80) bottom-right
      // nx=1 maps to right edge; y increases top->bottom along the right
      const px = x + nx * w;           // left at nx=0, right at nx=1
      const py = y + ny * h;           // top at ny=0, bottom at ny=1
      return [px, py];
    }
  }

  function canvasToSb(px, py) {
    const { x, y, w, h } = state.pitchRect;
    const v = state.sbView;
    let nx, ny;
    if (state.landscape) {
      const u = (px - x) / w; // 0..1 left->right is SB y
      const vpos = (py - y) / h; // 0..1 top->bottom is 1 - SB x
      ny = u;
      nx = 1 - vpos;
    } else {
      nx = (px - x) / w; // 0..1 left->right is SB x
      ny = (py - y) / h; // 0..1 top->bottom is SB y
    }
    const xSB = v.minX + nx * (v.maxX - v.minX);
    const ySB = v.minY + ny * (v.maxY - v.minY);
    return [xSB, ySB];
  }

  function drawLineSB(x1, y1, x2, y2) {
    const [px1, py1] = sbToCanvas(x1, y1);
    const [px2, py2] = sbToCanvas(x2, y2);
    strokeLine(px1, py1, px2, py2);
  }

  function drawRectSB(x0, y0, x1, y1) {
    const [p0x, p0y] = sbToCanvas(x0, y0);
    const [p1x, p1y] = sbToCanvas(x1, y1);
    const rx = Math.min(p0x, p1x), ry = Math.min(p0y, p1y);
    const rw = Math.abs(p1x - p0x), rh = Math.abs(p1y - p0y);
    ctx.strokeRect(rx, ry, rw, rh);
  }

  function drawSpotSB(xc, yc) {
    const [px, py] = sbToCanvas(xc, yc);
    drawSpot(px, py);
  }

  function drawPitch() {
    const { x, y, w, h } = state.pitchRect;
    const v = state.sbView;

    // Outer rectangle of the visible SB view
    ctx.save();
    const baseSize = state.landscape ? w : h; // use non-stretched axis as scale reference
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.lineWidth = Math.max(1, baseSize * 0.0026);
    ctx.shadowColor = 'rgba(110,231,255,0.25)';
    ctx.shadowBlur = Math.max(3, baseSize * 0.01);
    ctx.strokeRect(x, y, w, h);
    ctx.restore();

    // Internal lines within the view
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = Math.max(1, baseSize * 0.0016);
    ctx.shadowColor = 'transparent';

    // Draw the opponent goal line (x = 120) across y in [minY,maxY]
    drawLineSB(SB_LEN, v.minY, SB_LEN, v.maxY);

    // Draw 18-yard box near opponent goal: depth 18 from goal line, centered at y=40, width 44
    const boxDepth = 18; // along SB x
    const boxHalf = 44 / 2; // along SB y
    const y0 = 40 - boxHalf;
    const y1 = 40 + boxHalf;
    const x0 = SB_LEN - boxDepth;
    // Box rectangle (clipped to view by projection)
    drawRectSB(x0, y0, SB_LEN, y1);

    // 6-yard box
    const smallDepth = 6;
    const smallHalf = 20.15 / 2; // approx 6-yard box width 20.15
    const sy0 = 40 - smallHalf;
    const sy1 = 40 + smallHalf;
    const sx0 = SB_LEN - smallDepth;
    drawRectSB(sx0, sy0, SB_LEN, sy1);

    // Penalty spot moved to x = 108, y = 40
    drawSpotSB(108, 40);

    // Penalty arc (radius 9.15m from the spot), outside the penalty box
    // Compute angular range where points are outside the box (x <= SB_LEN - 18)
    (function drawPenaltyArc(){
      const cx = 108;
      const cy = 40;
      const r = 9.15;
      const xLine = SB_LEN - 18;
      const cosThresh = (xLine - cx) / r; // negative value
      const ang = Math.acos(Math.max(-1, Math.min(1, cosThresh)));
      const start = ang;
      const end = 2 * Math.PI - ang;
      const steps = 64;
      // Clip to pitch rect so arc doesn't draw outside the visible view border
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.beginPath();
      for (let i = 0; i <= steps; i++){
        const t = start + (end - start) * (i / steps);
        const xs = cx + r * Math.cos(t);
        const ys = cy + r * Math.sin(t);
        const [pxs, pys] = sbToCanvas(xs, ys);
        if (i === 0) ctx.moveTo(pxs, pys); else ctx.lineTo(pxs, pys);
      }
      ctx.stroke();
      ctx.restore();
    })();

    ctx.restore();
  }

  function drawSpot(px, py) {
    ctx.save();
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = 'rgba(110,231,255,0.35)';
    ctx.shadowBlur = 4;
    const baseSize = state.landscape ? state.pitchRect.w : state.pitchRect.h;
    const r = Math.max(2, baseSize * 0.0032); // slightly bigger penalty spot
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }

  function drawSquarePoint(px, py, size, glow=true) {
    ctx.save();
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = glow ? 'rgba(255,255,255,0.9)' : 'transparent';
    ctx.shadowBlur = glow ? 16 : 0;
    const s = size;
    ctx.fillRect(px - s/2, py - s/2, s, s);
    ctx.restore();
  }

  function getLabelFontPx(){
    // Use the same font sizing for both sequence-following labels and static labels
    const base = Math.min(state.pitchRect.w, state.pitchRect.h);
    return Math.max(9, base * 0.02);
  }

  function drawDatapoints() {
    const { w, h } = state.pitchRect;
    const now = performance.now();
    for (const p of state.points) {
      // Prefer SB coordinates if present
      let px, py;
      if (p.x != null && p.y != null) {
        [px, py] = sbToCanvas(p.x, p.y);
      } else {
        // Fallback: use normalized u,v mapping
        const { x, y } = state.pitchRect;
        if (state.landscape) {
          px = x + p.u * w;
          py = y + p.v * h;
        } else {
          px = x + p.v * w;
          py = y + p.u * h;
        }
      }

      // Age-based fade for both point and text
      const age = now - (p.born || now);
      const fullMs = 7000; // 7 seconds fully visible
      const fadeMs = 1500; // then fade out smoothly
      let alpha = 1;
      if (age > fullMs) {
        const t = Math.min(1, (age - fullMs) / fadeMs);
        alpha = 1 - (t*t*(3 - 2*t)); // smoothstep
      }
      if (alpha <= 0) continue;

      // Small glowing white square
      ctx.save();
      ctx.globalAlpha = alpha;
      drawSquarePoint(px, py, Math.max(4, Math.min(w, h) * 0.01), true);
      ctx.restore();

      // Stacked stats label: xG, (x,y), distance
      const finalFont = getLabelFontPx();
      const lines = p.lines || [`${p.xg.toFixed(2)} xG`];
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = `600 ${finalFont}px Orbitron, sans-serif`;
      ctx.fillStyle = '#e9f1ff';
      ctx.shadowColor = 'rgba(110,231,255,0.2)';
      ctx.shadowBlur = 5;
      // measure max width
      let maxW = 0;
      for (const s of lines) { const m = ctx.measureText(s); if (m.width > maxW) maxW = m.width; }
      const baseOffset = Math.max(6, Math.min(w, h) * 0.008);
      // Prefer saved side (from the line direction at creation) to keep text off the line
      const preferLeft = p.side === 'left';
      let baseX = preferLeft ? (px - maxW - baseOffset) : (px + baseOffset);
      // Edge-aware fallback flips if going off-screen
      if (baseX < 6) baseX = px + baseOffset;
      if (baseX + maxW > state.width - 6) baseX = px - maxW - baseOffset;
      let baseY = py - baseOffset;
      if (baseY - finalFont < 6) baseY = py + Math.max(finalFont, Math.min(w, h) * 0.035);
      const lh = finalFont * 1.18;

      // Backdrop to prevent obstruction by pitch lines
      const pad = Math.max(2, finalFont * 0.3);
      const bgX = baseX - pad;
      const bgY = (baseY - finalFont) - pad; // approx top of first line box
      const bgW = maxW + pad * 2;
      const bgH = lh * lines.length + pad * 2;
      ctx.save();
      ctx.shadowColor = 'transparent';
      // Match the soccer field backdrop color (same dark tone) instead of black
      ctx.fillStyle = 'rgba(11,15,20,0.85)';
      ctx.fillRect(bgX, bgY, bgW, bgH);
      ctx.restore();

      for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], baseX, baseY + i * lh);
      }
      ctx.restore();
    }
  }

  // Sequence renderer with smooth retract, reveal (first time only), and finalize
  function renderSequence(now) {
    const seq = state.seq;
    if (!seq) return false;
    const elapsed = now - seq.t0;

    // Phase timings
    const lineStart = 1000; // ms: pause with square
    const lineDur = 1000;   // ms: draw line out
    const textStart = lineStart + lineDur; // 2000ms
    const textDur = 900;    // ms: typewriter
    const retractStart = textStart + textDur; // 2900ms
    const retractDur = 800; // ms: retract line
    const revealStart = retractStart + retractDur; // 3700ms
    const revealDur = 1400; // ms: circular reveal of pitch from click point (first time only)
    const defocusDur = 450; // ms: gently transition spot size/glow to static (first time only)

    // Easing helpers
    const clamp01 = (t) => Math.max(0, Math.min(1, t));
    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
    const easeInCubic = (t) => t * t * t;
    const easeInOutCubic = (t) => t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;

    clear();

    // For subsequent clicks, keep the pitch visible throughout the sequence
    if (!seq.doReveal) {
      drawPitch();
      drawDatapoints();
    }

    // Always draw the square point during the sequence
    drawSquarePoint(seq.px, seq.py, 10, true);

    // Compute current line end based on phase
    let lineProgress = 0; // 0..1 extending
    if (elapsed >= lineStart) {
      if (elapsed < retractStart) {
        lineProgress = easeOutCubic(clamp01((elapsed - lineStart) / lineDur));
      } else {
        // Retraction: from 1 down to 0
        const r = easeInCubic(clamp01((elapsed - retractStart) / retractDur));
        lineProgress = 1 - r;
      }
    }

    // Draw line if any
    if (lineProgress > 0) {
      const x2 = seq.px + seq.dirX * seq.len * lineProgress;
      const y2 = seq.py + seq.dirY * seq.len * lineProgress;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 2;
      ctx.shadowColor = 'rgba(255,255,255,0.6)';
      ctx.shadowBlur = 10;
      strokeLine(seq.px, seq.py, x2, y2);
      ctx.restore();
      seq.anchorX = x2; // current tip for label attachment
      seq.anchorY = y2;
    } else {
      seq.anchorX = seq.px;
      seq.anchorY = seq.py;
    }

    // Render stacked stats: xG, (x,y), distance — follows the line tip and does NOT vanish on retract
    if (elapsed >= textStart) {
      ctx.save();
      const stackFont = seq.stackFont || getLabelFontPx();
      ctx.font = `600 ${stackFont}px Orbitron, sans-serif`;
      ctx.fillStyle = '#e9f1ff';
      ctx.shadowColor = 'rgba(110,231,255,0.25)';
      ctx.shadowBlur = 6;

      const offset = 8;
      // Measure max line width to choose side
      let maxW = 0;
      for (const s of seq.lines) {
        const m = ctx.measureText(s);
        if (m.width > maxW) maxW = m.width;
      }
      // Prefer side based on line direction: if line goes right, put text on right; if it goes left, put text on left
      let baseX = (seq.dirX >= 0) ? (seq.anchorX + offset) : (seq.anchorX - maxW - offset);
      // Edge-aware fallback: if chosen side goes off-screen, flip to the other side
      if (baseX < 6) baseX = seq.anchorX + offset;
      if (baseX + maxW > state.width - 6) baseX = seq.anchorX - maxW - offset;
      let baseY = seq.anchorY - 8;
      if (seq.anchorY < 20) baseY = seq.anchorY + 24;

      // Staggered fade-in per line, but keep fully visible thereafter
      const per = textDur / Math.max(1, seq.lines.length);
      const lh = stackFont * 1.18;

      // Compute visibility alpha to use for a single backdrop behind the whole stack
      let visAlpha = 0;
      const lineAlphas = new Array(seq.lines.length).fill(0);
      for (let i = 0; i < seq.lines.length; i++) {
        const tIn = clamp01((elapsed - textStart - i * per) / per);
        const a = easeInOutCubic(tIn);
        lineAlphas[i] = a;
        if (a > visAlpha) visAlpha = a;
      }

      // Draw small black backdrop so pitch lines don't obstruct text
      const pad = Math.max(2, stackFont * 0.3);
      const bgX = baseX - pad;
      const bgY = (baseY - stackFont) - pad; // approximate top
      // measure max width again to ensure consistency
      let maxW2 = 0; for (const s of seq.lines) { const m = ctx.measureText(s); if (m.width > maxW2) maxW2 = m.width; }
      const bgW = maxW2 + pad * 2;
      const bgH = lh * seq.lines.length + pad * 2;
      ctx.save();
      ctx.globalAlpha = visAlpha; // fade backdrop in with text
      ctx.shadowColor = 'transparent';
      // Match the soccer field backdrop color (same dark tone) instead of black
      ctx.fillStyle = 'rgba(11,15,20,0.85)';
      ctx.fillRect(bgX, bgY, bgW, bgH);
      ctx.restore();

      // Draw lines with their individual alphas on top of the backdrop
      for (let i = 0; i < seq.lines.length; i++) {
        const alphaIn = lineAlphas[i];
        if (alphaIn <= 0) continue;
        ctx.save();
        ctx.globalAlpha = alphaIn;
        ctx.fillText(seq.lines[i], baseX, baseY + i * lh);
        ctx.restore();
      }
      ctx.restore();
    }

    // Helper: compute static point size exactly like drawDatapoints
    const staticPointSize = Math.max(4, Math.min(state.pitchRect.w, state.pitchRect.h) * 0.01);

    // First click: reveal-from-darkness with circular clip; subsequent clicks skip this block
    if (seq.doReveal && elapsed >= revealStart) {
      const t = clamp01((elapsed - revealStart) / revealDur);
      const ease = easeOutCubic(t);
      const { x, y, w, h } = state.pitchRect;
      // radius to farthest corner ensures full reveal
      const maxR = Math.max(
        Math.hypot(seq.px - x, seq.py - y),
        Math.hypot(seq.px - (x + w), seq.py - y),
        Math.hypot(seq.px - x, seq.py - (y + h)),
        Math.hypot(seq.px - (x + w), seq.py - (y + h))
      );
      const r = maxR * ease;

      // Clip/reveal pitch
      ctx.save();
      ctx.beginPath();
      ctx.arc(seq.px, seq.py, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalAlpha = 0.2 + 0.8 * ease;
      drawPitch();
      // Also show any existing points (none on first click typically)
      drawDatapoints();
      ctx.restore();

      // Current point: during reveal keep at size 10, but towards end start easing to static size
      const endEaseThreshold = 0.85; // start blending near end
      let curSize = 10;
      if (ease > endEaseThreshold) {
        const k = (ease - endEaseThreshold) / (1 - endEaseThreshold);
        const blend = easeInOutCubic(clamp01(k));
        curSize = 10 + (staticPointSize - 10) * blend;
      }
      drawSquarePoint(seq.px, seq.py, curSize, true);
    }

    // Finalization logic
    if (!seq.doReveal) {
      // Subsequent clicks: end right after retract completes, commit point, keep everything on screen
      if (elapsed >= retractStart + retractDur) {
        if (!seq.committed) {
          state.points.push({ u: seq.u, v: seq.v, xg: seq.xg, x: seq.x, y: seq.y, lines: seq.lines, side: seq.side, born: performance.now() });
          seq.committed = true;
        }
        state.seq = null;
      }
    } else {
      // First click: after reveal completes, gently defocus then finish
      const endOfReveal = revealStart + revealDur;
      if (elapsed >= endOfReveal) {
        // Commit the point at the end of reveal so it exists in the static list
        if (!seq.committed) {
          state.points.push({ u: seq.u, v: seq.v, xg: seq.xg, x: seq.x, y: seq.y, lines: seq.lines, side: seq.side, born: performance.now() });
          seq.committed = true;
        }

        const tDefocus = clamp01((elapsed - endOfReveal) / defocusDur);
        const d = easeInOutCubic(tDefocus);

        // Draw full scene (no clip), with current point easing to static size/glow
        clear();
        drawPitch();
        drawDatapoints();

        // Overdraw the just-added current point with an easing size to match static exactly at t=1
        const curSize = 10 + (staticPointSize - 10) * d;
        const glowNow = 16; // keep consistent with static drawSquarePoint glow
        ctx.save();
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(255,255,255,0.9)';
        ctx.shadowBlur = glowNow;
        const s = curSize;
        ctx.fillRect(seq.px - s/2, seq.py - s/2, s, s);
        ctx.restore();

        // End when defocus finished
        if (tDefocus >= 1) {
          state.initialized = true;
          state.seq = null;
          if (!state.firstSequenceCompleted) {
            state.firstSequenceCompleted = true;
            maybeShowScrollCue();
          }
        }
      }
    }
    return true;
  }

  function draw(time) {
    const now = time || performance.now();

    if (state.seq) {
      renderSequence(now);
      return; // pitch hidden during sequence
    }

    clear();
    if (state.initialized) {
      drawPitch();
      drawDatapoints();
    }
  }

  function animate(time) {
    draw(time);
    requestAnimationFrame(animate);
  }

  // ------ Scroll transition to bio ------
  const root = document.documentElement;
  function setVar(name, value) { root.style.setProperty(name, value); }

  function setAppPointer(enabled) {
    const app = document.getElementById('app');
    if (!app) return;
    app.style.pointerEvents = enabled ? 'auto' : 'none';
  }

  function setCueVisible(show) {
    if (!scrollCue) return;
    if (show) scrollCue.classList.add('show');
    else scrollCue.classList.remove('show');
  }

  function maybeShowScrollCue() {
    if (!state.firstSequenceCompleted || !scrollCue) return;
    if (window.scrollY > 8) return;
    // slight delay to feel intentional
    setTimeout(() => setCueVisible(true), 250);
  }

  let scrollRaf = false;
  function applyScrollEffects() {
    scrollRaf = false;
    // Simple behavior: once the user scrolls a bit, hide the cue.
    const scrolled = window.scrollY > 10;
    if (scrolled) setCueVisible(false);
    // Show citation only after the user has scrolled a bit
    if (citationEl) {
      if (scrolled) citationEl.classList.add('show');
      else citationEl.classList.remove('show');
    }
  }

  function onScroll() {
    if (!scrollRaf) {
      scrollRaf = true;
      requestAnimationFrame(applyScrollEffects);
    }
  }

  function canvasToPitchUV(px, py) {
    const { x, y, w, h } = state.pitchRect;
    // Clamp to pitch rect, then convert to normalized (of the view rect axes)
    const cx = Math.max(x, Math.min(x + w, px));
    const cy = Math.max(y, Math.min(y + h, py));
    if (state.landscape) {
      return { u: (cx - x) / w, v: (cy - y) / h };
    } else {
      // In portrait, u follows vertical (top->bottom), v horizontal (left->right)
      return { u: (cy - y) / h, v: (cx - x) / w };
    }
  }

  function clampToViewSB(xSB, ySB) {
    const v = state.sbView;
    const xc = Math.max(v.minX, Math.min(v.maxX, xSB));
    const yc = Math.max(v.minY, Math.min(v.maxY, ySB));
    return [xc, yc];
  }

  function handlePointer(evt) {
    // Ignore if sequence running
    if (state.seq) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = (evt.clientX ?? (evt.touches && evt.touches[0].clientX));
    const clientY = (evt.clientY ?? (evt.touches && evt.touches[0].clientY));
    if (clientX == null || clientY == null) return;
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    if (promptEl) {
      // hide the prompt immediately on first actual interaction
      promptEl.style.display = 'none';
    }

    // Prepare sequence parameters
    const { u, v } = canvasToPitchUV(px, py);
    // Compute SB coordinates for this click and clamp to current view
    let [sx, sy] = canvasToSb(px, py);
    ;[sx, sy] = clampToViewSB(sx, sy);
    // Compute xG using the trained model if available; fall back to a deterministic tiny value if not yet loaded
    let xg = 0.0;
    if (xgPredictor && typeof xgPredictor.predict === 'function') {
      try {
        xg = xgPredictor.predict(sx, sy);
      } catch (e) {
        console.error('xG prediction error:', e);
        xg = 0.0;
      }
    } else {
      // fallback: use a simple heuristic based on distance to goal rather than random, until model loads
      const d = Math.hypot(SB_LEN - sx, 40 - sy);
      xg = Math.max(0, Math.min(1, 1 - d / 40));
    }

    // Choose a direction for the line so that label tends to stay on-screen
    const toRight = px < state.width * 0.6;
    const toUp = py > state.height * 0.6 ? true : false;
    const angle = Math.atan2(toUp ? -1 : -0.2, toRight ? 1 : -1); // slight upward bias
    const len = Math.max(60, Math.min(state.width, state.height) * 0.18);

    const doReveal = !state.firstSequenceCompleted; // only the first click reveals from darkness
    if (doReveal) state.initialized = false; // hide pitch only for the first suspense sequence

    // Precompute label content
    const initFont = Math.max(12, Math.min(state.width, state.height) * 0.035);
    const finalFont = Math.max(10, Math.min(state.width, state.height) * 0.024);
    const xStr = sx.toFixed(1);
    const yStr = sy.toFixed(1);
    const dist = Math.hypot(SB_LEN - sx, 40 - sy);
    const text = `${xg.toFixed(2)} xG · (x=${xStr}, y=${yStr}) · d=${dist.toFixed(1)}m`;
    const lines = [
      `${xg.toFixed(2)} xG`,
      `x=${xStr}, y=${yStr}`,
      `d=${dist.toFixed(1)}m`
    ];
    // Measure with final font to decide safe final placement
    ctx.save();
    ctx.font = `600 ${finalFont}px Orbitron, sans-serif`;
    const metrics = ctx.measureText(text);
    ctx.restore();
    const baseOffset = Math.max(6, Math.min(state.width, state.height) * 0.008);
    let finalTx = px + baseOffset;
    let finalTy = py - baseOffset;
    if (finalTx + metrics.width > state.width - 6) finalTx = px - metrics.width - baseOffset;
    if (finalTy - finalFont < 6) finalTy = py + Math.max(finalFont, Math.min(state.width, state.height) * 0.035);

    const dirX = Math.cos(angle);
    const dirY = Math.sin(angle);
    const side = dirX >= 0 ? 'right' : 'left';

    state.seq = {
      t0: performance.now(),
      px, py,
      dirX,
      dirY,
      len,
      text,
      lines,
      stackFont: Math.max(9, Math.min(state.width, state.height) * 0.02),
      u, v, xg,
      x: sx, y: sy, // SB coordinates within view
      side,
      committed: false,
      anchorX: px, anchorY: py,
      initFont,
      finalFont,
      finalTx,
      finalTy,
      doReveal
    };
  }

  // Event listeners
  window.addEventListener('resize', onResize);
  window.addEventListener('orientationchange', onResize);
  window.addEventListener('scroll', onScroll, { passive: true });
  canvas.addEventListener('pointerdown', handlePointer, { passive: true });
  canvas.addEventListener('touchstart', handlePointer, { passive: true });

  // Init
  onResize();
  startPromptTyping();
  // initialize scroll vars (in case page loads scrolled)
  applyScrollEffects();
  requestAnimationFrame(animate);
})();
