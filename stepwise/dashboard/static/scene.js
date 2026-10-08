/* The hero visualization: a neural network with a signal running through it,
 * and what a corruption does to that signal.
 *
 * This is not decoration — it dramatizes the project's central question. A
 * layered network sits behind the headline; a "command" propagates forward,
 * layer by layer, input -> output, lighting the units it passes and sending
 * bright pulses down the edges between layers (the forward pass).
 *
 * Periodically one unit is "corrupted" (flashes red) as the signal reaches it:
 *
 *   - corrupt a REASONING unit (a hidden layer) -> the signal through it fizzles
 *     and the OUTPUT does not move. "IGNORES": that unit wasn't load-bearing.
 *   - corrupt the ANSWER unit (the output)      -> the output flips.
 *     "FOLLOWS": corrupting the unit that states the answer changes the answer —
 *     the answer-step tautology, seen as signal flow.
 *
 * Which corruptions flip the output is nudged by the real run via setFinding(),
 * so the animation stays honest about the numbers below it. three.js is loaded
 * as a UMD global from cdnjs.
 */
(function () {
  "use strict";

  const host = document.getElementById("scene");
  const caption = document.getElementById("scene-caption");
  if (!host || typeof THREE === "undefined") return;

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Palette pulled from the page's own tokens, so the scene matches the theme.
  function token(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  // three.js r128's Color can't parse oklch(), so these are hex equivalents of
  // the page's --accent / --accent-up tokens (kept in sync by hand).
  // Matte teal-on-paper: a calm instrument diagram, not a neon display.
  const COL = {
    node:    new THREE.Color("#1C7A6D"),      // resting unit (deep teal)
    lit:     new THREE.Color("#41B0A0"),      // a unit the signal is passing through
    answer:  new THREE.Color("#0E5A50"),
    follows: new THREE.Color("#C0432A"),      // rust — the answer moved with the error
    ignores: new THREE.Color("#B9892F"),
    corrupt: new THREE.Color("#C0432A"),
    edge:    new THREE.Color("#CFC8B6"),      // warm grey lines on paper
    edgeHot: new THREE.Color("#1C7A6D"),
    signal:  new THREE.Color("#1C7A6D"),
  };

  const scene = new THREE.Scene();
  const W = () => host.clientWidth || 900;
  const H = () => host.clientHeight || 560;
  const FOV = 42;
  const camera = new THREE.PerspectiveCamera(FOV, W() / H(), 0.1, 100);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W(), H());
  // r128 defaults to linear output, which washes saturated colours out on a
  // light background; sRGB output gives the nodes their true depth.
  if ("outputColorSpace" in renderer && THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace;
  else if ("outputEncoding" in renderer && THREE.sRGBEncoding) renderer.outputEncoding = THREE.sRGBEncoding;
  host.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const key = new THREE.PointLight(0xffffff, 34, 120);
  key.position.set(6, 7, 11);
  scene.add(key);
  const rim = new THREE.PointLight(0x9db4ff, 12, 120);
  scene.add(rim);
  rim.position.set(-7, -3, 6);

  // --- Build the network: a few layers of units, fully connected. ----------
  const LAYERS = [4, 6, 6, 3];           // input, hidden, hidden, output
  const L = LAYERS.length;
  const X_SPAN = 15.4, Y_GAP = 1.16;
  const group = new THREE.Group();
  scene.add(group);

  // The "answer" unit: the top node of the output layer, drawn larger.
  const ANSWER_LAYER = L - 1, ANSWER_INDEX = 0;

  const layers = [];                      // layers[l] = array of node objects
  for (let l = 0; l < L; l++) {
    const n = LAYERS[l];
    const x = -X_SPAN / 2 + (X_SPAN * l) / (L - 1);
    const col = [];
    for (let i = 0; i < n; i++) {
      const isAnswer = l === ANSWER_LAYER && i === ANSWER_INDEX;
      const r = isAnswer ? 0.5 : (l === 0 ? 0.3 : 0.36);
      const geo = new THREE.IcosahedronGeometry(r, isAnswer ? 2 : 1);
      const base = (l === ANSWER_LAYER) ? COL.answer : COL.node;
      // Flat/unlit: renders the exact on-brand colour with no lighting washout.
      // "Lit", corrupt and flip states are driven by colour + scale below.
      const mat = new THREE.MeshBasicMaterial({ color: base.clone() });
      const mesh = new THREE.Mesh(geo, mat);
      const y = (i - (n - 1) / 2) * Y_GAP;
      const z = Math.sin(i * 1.7 + l) * 0.35;      // gentle depth
      mesh.position.set(x, y, z);
      group.add(mesh);
      // A whisper-soft aura behind each node — just enough to give the matte
      // diagram some depth on paper. Normal blending, low opacity (no glow).
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(r * 1.9, 16, 16),
        new THREE.MeshBasicMaterial({ color: base.clone(), transparent: true,
          opacity: 0.08, depthWrite: false })
      );
      halo.position.copy(mesh.position);
      group.add(halo);
      col.push({ mesh, mat, base, isAnswer, lit: 0, halo });
    }
    layers.push(col);
  }

  // Edges: every node in layer l to every node in l+1. One LineSegments mesh
  // with per-vertex colour so an active gap can brighten cheaply.
  const edgeGaps = [];                    // edgeGaps[gap] = [{a, b, vi}]
  const positions = [];
  const colors = [];
  let vi = 0;
  for (let l = 0; l < L - 1; l++) {
    const gap = [];
    for (const a of layers[l]) {
      for (const b of layers[l + 1]) {
        positions.push(a.mesh.position.x, a.mesh.position.y, a.mesh.position.z);
        positions.push(b.mesh.position.x, b.mesh.position.y, b.mesh.position.z);
        colors.push(COL.edge.r, COL.edge.g, COL.edge.b, COL.edge.r, COL.edge.g, COL.edge.b);
        gap.push({ a, b, vi });
        vi += 2;
      }
    }
    edgeGaps.push(gap);
  }
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  edgeGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const edgeMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.68 });
  const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
  group.add(edgeLines);
  const edgeColorAttr = edgeGeo.getAttribute("color");

  function setEdgeColor(vi, c, k) {
    // Lerp edge endpoints toward colour c by amount k, in place.
    for (let o = 0; o < 2; o++) {
      const idx = (vi + o) * 3;
      edgeColorAttr.array[idx]     += (c.r - edgeColorAttr.array[idx])     * k;
      edgeColorAttr.array[idx + 1] += (c.g - edgeColorAttr.array[idx + 1]) * k;
      edgeColorAttr.array[idx + 2] += (c.b - edgeColorAttr.array[idx + 2]) * k;
    }
  }

  // Travelling signal pulses: a small pool reused each gap.
  const PULSES = 18;
  const pulsePool = [];
  for (let i = 0; i < PULSES; i++) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 12, 12),
      new THREE.MeshBasicMaterial({ color: COL.signal, transparent: true, opacity: 0.95 })
    );
    m.visible = false;
    scene.add(m);
    pulsePool.push({ mesh: m, edge: null, fade: false });
  }

  // Halo behind the answer unit.
  const ans = layers[ANSWER_LAYER][ANSWER_INDEX];
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.82, 0.028, 8, 48),
    new THREE.MeshBasicMaterial({ color: COL.answer, transparent: true, opacity: 0.32 })
  );
  ring.position.copy(ans.mesh.position);
  scene.add(ring);

  // --- Finding: which corruptions flip the output. ------------------------
  // Default: only the answer unit flips (the tautology). If the real run shows
  // reasoning-step dependence, let a hidden unit sometimes flip too.
  let reasoningCanFlip = false;
  window.stepwiseScene = {
    setFinding(summary) {
      try {
        const res = summary.by_model_reasoning || [];
        reasoningCanFlip = res.some((r) => (r.dependency || 0) > 0.15);
      } catch (e) { /* keep default */ }
    },
  };
  if (window.__stepwiseSummary) window.stepwiseScene.setFinding(window.__stepwiseSummary);

  // --- Animation --------------------------------------------------------- //
  function setCaption(text, color) {
    if (!caption) return;
    caption.textContent = text;
    caption.style.color = color || "";
  }
  setCaption("forward pass · a signal runs input → output", token("--ink-3", "#777"));

  const clock = new THREE.Clock();
  let head = 0;           // fractional wavefront position in [0, L-1]
  const GAP_DUR = 0.95;   // seconds to cross one gap
  const PASS_PAUSE = 1.4; // pause between passes
  let passState = "run";  // run | pause
  let pauseUntil = 0;
  let activeGap = -1;     // gap whose pulses are in flight

  // Corruption for the current pass.
  let corruptLayer = -1, corruptIndex = -1, corruptFlips = false, resolved = false;

  function planCorruption() {
    // 70% of passes corrupt a hidden (reasoning) unit; 30% the answer unit.
    const hitAnswer = Math.random() < 0.3;
    if (hitAnswer) {
      corruptLayer = ANSWER_LAYER; corruptIndex = ANSWER_INDEX; corruptFlips = true;
    } else {
      corruptLayer = 1 + Math.floor(Math.random() * (L - 2));          // a hidden layer
      corruptIndex = Math.floor(Math.random() * LAYERS[corruptLayer]);
      corruptFlips = reasoningCanFlip && Math.random() < 0.4;
    }
    resolved = false;
  }
  planCorruption();

  function assignPulses(gap) {
    const edges = edgeGaps[gap];
    for (let i = 0; i < pulsePool.length; i++) {
      const p = pulsePool[i];
      const e = edges[Math.floor(Math.random() * edges.length)];
      p.edge = e;
      p.mesh.visible = true;
      p.mesh.scale.setScalar(1);
      p.mesh.material.color.copy(COL.signal);
    }
  }

  function node(l, i) { return layers[l][i]; }
  const corruptNode = () => (corruptLayer >= 0 ? node(corruptLayer, corruptIndex) : null);

  // The driver: a single rAF loop on a manual delta clock.
  let last = performance.now();
  // Subtle mouse parallax — the network leans toward the cursor.
  const mouse = { tx: 0, ty: 0, x: 0, y: 0 };
  addEventListener("pointermove", (e) => {
    mouse.tx = (e.clientX / innerWidth - 0.5) * 0.5;
    mouse.ty = (e.clientY / innerHeight - 0.5) * 0.3;
  });

  function loop(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = clock.getElapsedTime();

    // Idle motion + eased mouse parallax.
    mouse.x += (mouse.tx - mouse.x) * 0.05;
    mouse.y += (mouse.ty - mouse.y) * 0.05;
    group.rotation.y = Math.sin(t * 0.14) * 0.16 + mouse.x;
    group.rotation.x = Math.sin(t * 0.10) * 0.045 - mouse.y;
    ring.rotation.z = t * 0.5;
    ring.rotation.x = t * 0.28;

    // Advance the wavefront.
    if (passState === "run") {
      const prev = head;
      head += dt / GAP_DUR;
      const gap = Math.floor(head);
      // Entering a new gap: assign pulses for it, light the source layer.
      if (gap !== activeGap && gap < L - 1) {
        activeGap = gap;
        assignPulses(gap);
        for (const nd of layers[gap]) nd.lit = 1;      // source layer fires
      }
      // Reached the corrupt node's layer: flash it.
      // (handled by colour logic below)
      if (head >= L - 1) {
        // Arrived at output: resolve the pass.
        for (const nd of layers[L - 1]) nd.lit = 1;
        if (!resolved) resolvePass();
        passState = "pause";
        pauseUntil = t + PASS_PAUSE;
        for (const p of pulsePool) p.mesh.visible = false;
        activeGap = -1;
      }
    } else if (passState === "pause" && t >= pauseUntil) {
      // Reset for the next pass.
      head = 0; activeGap = -1;
      // restore output nodes to base before the next corruption
      for (const nd of layers[L - 1]) { nd.mat.color.copy(nd.base); nd.mesh.scale.setScalar(1); }
      planCorruption();
      passState = "run";
      setCaption("forward pass · a signal runs input → output", token("--ink-3", "#777"));
    }

    // Pulses in flight along the active gap.
    if (passState === "run" && activeGap >= 0) {
      const p = Math.min(1, head - activeGap);
      const gapEdges = edgeGaps[activeGap];
      // Brighten the active gap's edges toward hot.
      for (const e of gapEdges) setEdgeColor(e.vi, COL.edgeHot, 0.14);
      for (const pl of pulsePool) {
        if (!pl.edge) continue;
        const a = pl.edge.a.mesh.position, b = pl.edge.b.mesh.position;
        pl.mesh.position.lerpVectors(a, b, p);
        // A pulse leaving the corrupted reasoning unit fades out mid-edge.
        const fromCorrupt = pl.edge.a === corruptNode() && !corruptFlips;
        if (fromCorrupt) {
          pl.mesh.scale.setScalar(Math.max(0.001, 1 - p * 1.4));
          pl.mesh.material.color.copy(COL.ignores);
        }
      }
    }

    // Corruption node colouring: flash red once the wave reaches its layer.
    const cn = corruptNode();
    if (cn) {
      const reached = head >= corruptLayer - 0.15;
      if (reached && passState === "run") {
        cn.mat.color.lerp(COL.corrupt, 0.22);
        cn.mesh.scale.setScalar(1.35);
        if (caption && caption.dataset.state !== "corrupt") {
          caption.dataset.state = "corrupt";
          setCaption(corruptLayer === ANSWER_LAYER
            ? "corrupting the answer unit…"
            : `corrupting a reasoning unit · layer ${corruptLayer}…`, token("--ink-2", "#555"));
        }
      }
    }

    // Node colour + scale: resting = base; a passing signal brightens toward
    // COL.lit and swells the unit; corrupt/flip states are held separately.
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < LAYERS[l]; i++) {
        const nd = layers[l][i];
        nd.mesh.rotation.x = t * 0.25 + i;
        nd.mesh.rotation.y = t * 0.18 + l;
        const isCorrupt = nd === cn && head >= corruptLayer - 0.15;
        const isFlippedAns = nd.isAnswer && resolved && corruptFlips;
        if (!isCorrupt && !isFlippedAns) {
          nd.mat.color.copy(nd.base).lerp(COL.lit, Math.min(1, nd.lit) * 0.9);
          nd.mesh.scale.setScalar(1 + Math.min(1, nd.lit) * 0.32);
          nd.lit *= 0.95;
        }
        // Halo follows the node and blooms with its activity.
        if (nd.halo) {
          nd.halo.position.copy(nd.mesh.position);
          nd.halo.material.color.copy(nd.mat.color);
          const glow = nd.isAnswer ? 0.12 : 0.07;
          nd.halo.material.opacity = glow + Math.min(1, nd.lit) * 0.18;
          nd.halo.scale.setScalar(1 + Math.min(1, nd.lit) * 0.4);
        }
      }
    }
    for (let k = 0; k < edgeColorAttr.array.length; k += 3) {
      edgeColorAttr.array[k]     += (COL.edge.r - edgeColorAttr.array[k])     * 0.05;
      edgeColorAttr.array[k + 1] += (COL.edge.g - edgeColorAttr.array[k + 1]) * 0.05;
      edgeColorAttr.array[k + 2] += (COL.edge.b - edgeColorAttr.array[k + 2]) * 0.05;
    }
    edgeColorAttr.needsUpdate = true;

    rim.position.x = Math.sin(t * 0.3) * 7;
    renderer.render(scene, camera);
    if (!reduced) requestAnimationFrame(loop);
  }

  function resolvePass() {
    resolved = true;
    caption.dataset.state = "resolve";
    if (corruptFlips) {
      const a = ans;
      a.mat.color.copy(COL.follows);
      a.mesh.scale.setScalar(1.3);
      setCaption("FOLLOWS — the output moved with the error", "#8b6cff");
    } else {
      setCaption("IGNORES — the output held; that unit wasn't load-bearing",
                 token("--ignores", "#c99a3a"));
    }
    setTimeout(() => { if (caption) caption.dataset.state = ""; }, 50);
  }

  function fitCamera() {
    // Measure the host directly rather than through W()/H(): those fall back to
    // a default when the hero is hidden, and sizing the renderer to that default
    // is what used to leave the canvas stranded at 900x560 inside a wide hero.
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;            // still hidden — the observer calls us again
    const aspect = w / h;
    camera.aspect = aspect;
    const vHalf = Math.tan((FOV * Math.PI) / 180 / 2);
    const narrow = w < 760;
    const halfH = 3.7, halfW = narrow ? 8.4 : 10.4;   // world extents + margin
    const d = Math.max(halfH / vHalf, halfW / (vHalf * aspect));
    // Anchor the network's right edge just inside the frame so a wide viewport
    // cannot leave dead space beside it; on a narrow screen the copy stacks
    // above, so centre the whole net instead.
    const halfVisW = d * vHalf * aspect;
    const cx = narrow ? 0 : Math.min(0, X_SPAN / 2 + 0.9 - halfVisW);
    camera.position.set(cx, 0.2, d);
    camera.lookAt(cx, 0.1, 0);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  fitCamera();
  addEventListener("resize", fitCamera);
  // The hero is hidden when this script runs (views are revealed after the run
  // loads), so the first fit is a no-op. A ResizeObserver covers later reflows,
  // but it is not reliable for the display:none -> visible transition, so the
  // page also calls stepwiseScene.resize() whenever it reveals a view.
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => fitCamera()).observe(host);
  }
  window.stepwiseScene.resize = fitCamera;

  if (reduced) {
    // Reduced motion: one clean still frame, no loop, no corruption.
    setCaption("a neural network · corrupt a unit, watch the output", token("--ink-3", "#777"));
    for (const nd of layers[1]) { nd.mat.color.copy(COL.lit); nd.mesh.scale.setScalar(1.2); }
    renderer.render(scene, camera);
  } else {
    requestAnimationFrame(loop);
  }
})();
