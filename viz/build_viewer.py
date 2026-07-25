"""Build a self-contained hex-truchet game viewer (content-only HTML for the
Artifact tool: no <html>/<head>/<body> wrappers). Embeds game_data.json inline.
"""
import json, os

here = os.path.dirname(__file__)
with open(os.path.join(here, "game_data.json")) as f:
    data = json.load(f)

TEMPLATE = r"""<style>
:root {
  --bg:#0f1116; --panel:#171a22; --panel2:#1e222c; --hair:#242a36;
  --line:#333b49; --ink:#d7dae3; --muted:#828a99; --faint:#565e6d;
  --p0:#e5a94e; --p1:#4ec9c9;
  --cell:#1a1e28; --cell-line:#2b3240;
  /* Player A = warm ramp, Player B = cool ramp: hue family shows the owner,
     shade within it distinguishes individual loops. */
  --a0:#e5a94e; --a1:#ef8f5b; --a2:#e0587a; --a3:#d9c24a;
  --b0:#4ec9c9; --b1:#7c9cff; --b2:#5ecb8a; --b3:#59b6d6;
  --shadow: 0 1px 0 rgba(255,255,255,.03), 0 12px 40px -12px rgba(0,0,0,.6);
}
@media (prefers-color-scheme: light) {
  :root {
    --bg:#eef0ec; --panel:#ffffff; --panel2:#f6f7f4; --hair:#e2e5e0;
    --line:#d3d7d0; --ink:#242a30; --muted:#5f6771; --faint:#9aa0a8;
    --p0:#bd7716; --p1:#1a8f8f;
    --cell:#f7f8f5; --cell-line:#e2e6df;
    --a0:#c67f18; --a1:#d1622f; --a2:#cf3f63; --a3:#b08a12;
    --b0:#159090; --b1:#4d6fe0; --b2:#2f9e63; --b3:#2f83a8;
    --shadow: 0 1px 0 rgba(0,0,0,.02), 0 14px 40px -18px rgba(30,40,50,.35);
  }
}
:root[data-theme="dark"] {
  --bg:#0f1116; --panel:#171a22; --panel2:#1e222c; --hair:#242a36;
  --line:#333b49; --ink:#d7dae3; --muted:#828a99; --faint:#565e6d;
  --p0:#e5a94e; --p1:#4ec9c9; --cell:#1a1e28; --cell-line:#2b3240;
  --a0:#e5a94e; --a1:#ef8f5b; --a2:#e0587a; --a3:#d9c24a;
  --b0:#4ec9c9; --b1:#7c9cff; --b2:#5ecb8a; --b3:#59b6d6;
  --shadow: 0 1px 0 rgba(255,255,255,.03), 0 12px 40px -12px rgba(0,0,0,.6);
}
:root[data-theme="light"] {
  --bg:#eef0ec; --panel:#ffffff; --panel2:#f6f7f4; --hair:#e2e5e0;
  --line:#d3d7d0; --ink:#242a30; --muted:#5f6771; --faint:#9aa0a8;
  --p0:#bd7716; --p1:#1a8f8f; --cell:#f7f8f5; --cell-line:#e2e6df;
  --a0:#c67f18; --a1:#d1622f; --a2:#cf3f63; --a3:#b08a12;
  --b0:#159090; --b1:#4d6fe0; --b2:#2f9e63; --b3:#2f83a8;
  --shadow: 0 1px 0 rgba(0,0,0,.02), 0 14px 40px -18px rgba(30,40,50,.35);
}
* { box-sizing:border-box; }
.wrap {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink); background: var(--bg);
  min-height:100%; padding: clamp(16px,3vw,34px);
  display:flex; flex-direction:column; gap:20px;
}
.mono { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
header { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:12px 24px; }
.eyebrow { font-size:11px; letter-spacing:.22em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 6px; }
h1 { margin:0; font-size: clamp(22px,3.4vw,32px); letter-spacing:-.02em; font-weight:700; text-wrap:balance; }
.sub { color:var(--muted); font-size:13.5px; max-width:60ch; margin:6px 0 0; line-height:1.5; }
.rules { display:flex; flex-wrap:wrap; gap:6px; margin-top:2px; }
.tag { font-size:11px; color:var(--muted); background:var(--panel2); border:1px solid var(--hair); border-radius:999px; padding:3px 9px; }
.tag b { color:var(--ink); font-weight:600; }

.stage { display:grid; grid-template-columns: minmax(0,1fr) 268px; gap:20px; align-items:start; }
@media (max-width: 780px){ .stage { grid-template-columns:1fr; } }

.boardcard { background:var(--panel); border:1px solid var(--hair); border-radius:16px; padding:14px; box-shadow:var(--shadow); }
svg.board { width:100%; height:auto; display:block; }
.hexcell { fill:var(--cell); stroke:var(--cell-line); stroke-width:1; }
.hexcell.p0 { fill: color-mix(in srgb, var(--p0) 9%, var(--cell)); }
.hexcell.p1 { fill: color-mix(in srgb, var(--p1) 9%, var(--cell)); }
.hexcell.latest { stroke:var(--ink); stroke-width:2; }
.loopfill { stroke:none; opacity:.16; }
.arc { fill:none; stroke-linecap:round; }
.arc.run0 { stroke: color-mix(in srgb, var(--p0) 62%, var(--faint)); stroke-width:3.2; opacity:.62; }
.arc.run1 { stroke: color-mix(in srgb, var(--p1) 62%, var(--faint)); stroke-width:3.2; opacity:.62; }
.arc.loop { stroke-width:4.2; }
.arc.pop { animation: pop .5s ease-out; }
@keyframes pop { from { stroke-width:8; opacity:.4; } to {} }
@media (prefers-reduced-motion: reduce){ .arc.pop { animation:none; } }

.rail { display:flex; flex-direction:column; gap:12px; }
.card { background:var(--panel); border:1px solid var(--hair); border-radius:14px; padding:13px 14px; box-shadow:var(--shadow); }
.scores { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.score { background:var(--panel2); border:1px solid var(--hair); border-radius:12px; padding:11px 12px; position:relative; overflow:hidden; }
.score .who { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; display:flex; align-items:center; gap:7px; }
.dot { width:9px; height:9px; border-radius:50%; }
.score.s0 .dot { background:var(--p0); } .score.s1 .dot { background:var(--p1); }
.score .val { font-size:30px; font-weight:700; margin-top:4px; letter-spacing:-.02em; }
.score.turn { outline:2px solid color-mix(in srgb, var(--ink) 30%, transparent); }
.card h3 { margin:0 0 9px; font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.move { display:flex; align-items:baseline; justify-content:space-between; gap:8px; }
.move .lead { font-size:13.5px; color:var(--ink); }
.move .gain { font-size:22px; font-weight:700; }
.move .gain.zero { color:var(--faint); }
.loomlist { display:flex; flex-direction:column; gap:7px; }
.looprow { display:flex; align-items:center; gap:9px; font-size:12.5px; }
.looprow .sw { width:11px; height:11px; border-radius:3px; flex:none; }
.looprow .len { color:var(--muted); margin-left:auto; }
.empty { color:var(--faint); font-size:12.5px; font-style:italic; }

.transport { display:flex; align-items:center; gap:14px; background:var(--panel); border:1px solid var(--hair); border-radius:14px; padding:12px 16px; box-shadow:var(--shadow); flex-wrap:wrap; }
.btn { background:var(--panel2); color:var(--ink); border:1px solid var(--line); border-radius:9px; height:38px; min-width:44px; padding:0 12px; font-size:14px; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; gap:6px; }
.btn:hover { border-color:var(--ink); }
.btn.play { min-width:96px; font-weight:600; }
.btn:focus-visible, input:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
.scrub { flex:1 1 220px; display:flex; align-items:center; gap:12px; }
input[type=range]{ -webkit-appearance:none; appearance:none; width:100%; height:5px; border-radius:3px; background:var(--line); cursor:pointer; }
input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:var(--ink); border:2px solid var(--panel); }
input[type=range]::-moz-range-thumb{ width:16px; height:16px; border-radius:50%; background:var(--ink); border:2px solid var(--panel); }
.readout { font-size:13px; color:var(--muted); white-space:nowrap; }
.readout b { color:var(--ink); }
.speed { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }
select { background:var(--panel2); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:6px 8px; font-size:12.5px; }
</style>

<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">Hex Truchet · Stage 3 environment</p>
      <h1>Loops, drawn one tile at a time</h1>
      <p class="sub">A recorded game on the radius-3 board. Two greedy players alternate placing arc tiles; whenever the arcs seal a closed loop, its player scores the number of cells it encloses. Loops are tinted by who closed them — <span style="color:var(--p0);font-weight:600">warm = Player A</span>, <span style="color:var(--p1);font-weight:600">cool = Player B</span>. Scrub to watch loops form.</p>
      <div class="rules">
        <span class="tag"><b>37</b> cells</span>
        <span class="tag">deck <b>12:25</b></span>
        <span class="tag"><b>hand</b> of 3</span>
        <span class="tag"><b>adjacency</b> placement</span>
        <span class="tag">score = <b>area</b></span>
      </div>
    </div>
  </header>

  <div class="stage">
    <div class="boardcard">
      <svg class="board" id="board" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="rail">
      <div class="scores">
        <div class="score s0" id="sc0"><div class="who"><span class="dot"></span>Player A</div><div class="val mono" id="v0">0</div></div>
        <div class="score s1" id="sc1"><div class="who"><span class="dot"></span>Player B</div><div class="val mono" id="v1">0</div></div>
      </div>
      <div class="card">
        <h3>This move</h3>
        <div class="move"><span class="lead" id="movelead">—</span><span class="gain mono zero" id="movegain">+0</span></div>
      </div>
      <div class="card">
        <h3>Closed loops <span class="mono" id="loopcount" style="color:var(--ink)"></span></h3>
        <div class="loomlist" id="looplist"><div class="empty">none yet</div></div>
      </div>
    </div>
  </div>

  <div class="transport">
    <button class="btn" id="prev" aria-label="Previous move">‹</button>
    <button class="btn play" id="play">▶ Play</button>
    <button class="btn" id="next" aria-label="Next move">›</button>
    <div class="scrub">
      <input type="range" id="slider" min="0" max="37" value="0" step="1" aria-label="Move">
      <span class="readout mono">move <b id="stepnum">0</b>/37</span>
    </div>
    <label class="speed">speed
      <select id="speed"><option value="700">0.5×</option><option value="380" selected>1×</option><option value="180">2×</option><option value="90">4×</option></select>
    </label>
  </div>
</div>

<script>
const GAME = __GAME_DATA__;
// Player A (warm) and Player B (cool) ramps: hue family = who scored the loop,
// shade within = which loop. loopColor(owner, k) picks the k-th shade.
const RAMP = [['--a0','--a1','--a2','--a3'], ['--b0','--b1','--b2','--b3']];
function loopColor(owner, k){ const r = RAMP[owner]; return r[k % r.length]; }
const SVGNS = 'http://www.w3.org/2000/svg';
const eo = GAME.edge_off, cells = GAME.cells, steps = GAME.steps;

function cvar(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function span(a,b){ let d=Math.abs(a-b)%6; return Math.min(d,6-d); }

// tangent-continuous circular arc between two edge-midpoints of one cell.
// Sample explicit points along the true arc (short angular direction around
// the real circle center) rather than emit an SVG "A" command -- "A"
// re-derives its own center from the radius+flags and can pick the reflected
// center, throwing the arc OUTSIDE the cell. Sampling guarantees the intended
// inside arc, with radial tangents at both ends so loops flow across cells.
function arcPath(cx,cy,ea,eb){
  const Pa=[cx+eo[ea][0], cy+eo[ea][1]], Pb=[cx+eo[eb][0], cy+eo[eb][1]];
  const line = () => `M${Pa[0].toFixed(2)},${Pa[1].toFixed(2)} L${Pb[0].toFixed(2)},${Pb[1].toFixed(2)}`;
  if(span(ea,eb)===3) return line();
  const Ta=[-eo[ea][1],eo[ea][0]], Tb=[-eo[eb][1],eo[eb][0]];
  const det = Ta[0]*(-Tb[1]) - (-Tb[0])*Ta[1];
  if(Math.abs(det)<1e-6) return line();
  const rx=Pb[0]-Pa[0], ry=Pb[1]-Pa[1];
  const s = (rx*(-Tb[1]) - (-Tb[0])*ry)/det;
  const Oc=[Pa[0]+s*Ta[0], Pa[1]+s*Ta[1]];
  const r = Math.hypot(Oc[0]-Pa[0], Oc[1]-Pa[1]);
  const a0 = Math.atan2(Pa[1]-Oc[1], Pa[0]-Oc[0]);
  let d = Math.atan2(Pb[1]-Oc[1], Pb[0]-Oc[0]) - a0;
  while(d>Math.PI) d-=2*Math.PI; while(d<-Math.PI) d+=2*Math.PI;
  const N=18; let p=`M${Pa[0].toFixed(2)},${Pa[1].toFixed(2)}`;
  for(let i=1;i<=N;i++){ const a=a0+d*i/N;
    p+=` L${(Oc[0]+r*Math.cos(a)).toFixed(2)},${(Oc[1]+r*Math.sin(a)).toFixed(2)}`; }
  return p;
}
function hexPoints(cx,cy){
  const R = GAME.L/Math.sqrt(3); let p=[];
  for(let k=0;k<6;k++){ const a=(30+60*k)*Math.PI/180; p.push(`${(cx+R*Math.cos(a)).toFixed(2)},${(cy+R*Math.sin(a)).toFixed(2)}`); }
  return p.join(' ');
}

const board = document.getElementById('board');
board.setAttribute('viewBox', `0 0 ${GAME.W} ${GAME.H}`);

let state = 0;              // number of placements shown (0..37)
let lastRendered = -1;

function render(s, animate){
  state = s;
  const cur = s>0 ? steps[s-1] : null;
  const loops = cur ? cur.loops : [];
  // colour each loop from its OWNER's ramp; index within the owner's own loops
  // so same-player loops still get distinct shades. Map arcs+cells -> css var.
  const loopArc = new Map(); const cellLoop = new Map();
  const perOwner = [0,0];
  loops.forEach(lp=>{
    const k = perOwner[lp.owner]++;
    const cvarName = loopColor(lp.owner, k);
    lp._c = cvarName;
    lp.arcs.forEach(a=>{ const key=`${a[0]}-${Math.min(a[1],a[2])}-${Math.max(a[1],a[2])}`; loopArc.set(key,cvarName); });
    lp.cells.forEach(c=> cellLoop.set(c,cvarName));
  });
  const latestCell = cur ? cur.cell : -1;
  const owner = {};
  for(let j=0;j<s;j++) owner[steps[j].cell]=steps[j].player;

  let out = '';
  // shaded enclosed cells (under everything)
  // hex cells
  for(const cell of cells){
    const [cx,cy]=cell.center; const cls=['hexcell'];
    if(owner[cell.idx]!==undefined) cls.push('p'+owner[cell.idx]);
    if(cell.idx===latestCell) cls.push('latest');
    out += `<polygon class="${cls.join(' ')}" points="${hexPoints(cx,cy)}"/>`;
    if(cellLoop.has(cell.idx)){
      out += `<polygon class="loopfill" points="${hexPoints(cx,cy)}" fill="var(${cellLoop.get(cell.idx)})"/>`;
    }
  }
  // arcs
  for(let j=0;j<s;j++){
    const st=steps[j]; const [cx,cy]=cells[st.cell].center;
    for(const [ea,eb] of st.arcs){
      const key=`${st.cell}-${Math.min(ea,eb)}-${Math.max(ea,eb)}`;
      if(loopArc.has(key)){
        const col=`var(${loopArc.get(key)})`;
        const pop = (animate && st.cell===latestCell)?' pop':'';
        out += `<path class="arc loop${pop}" style="stroke:${col}" d="${arcPath(cx,cy,ea,eb)}"/>`;
      } else {
        const pop = (animate && st.cell===latestCell)?' pop':'';
        out += `<path class="arc run${st.player}${pop}" d="${arcPath(cx,cy,ea,eb)}"/>`;
      }
    }
  }
  board.innerHTML = out;

  // rail
  const sc=[0,0]; if(cur){ sc[0]=cur.score[0]; sc[1]=cur.score[1]; }
  document.getElementById('v0').textContent=sc[0];
  document.getElementById('v1').textContent=sc[1];
  const nextPlayer = s%2; // who acts on the upcoming move
  document.getElementById('sc0').classList.toggle('turn', s<37 && nextPlayer===0);
  document.getElementById('sc1').classList.toggle('turn', s<37 && nextPlayer===1);

  const ml=document.getElementById('movelead'), mg=document.getElementById('movegain');
  if(cur){ ml.textContent=`Player ${cur.player===0?'A':'B'} placed tile ${cur.tile}`;
    mg.textContent=(cur.gained>0?'+':'')+cur.gained; mg.classList.toggle('zero',cur.gained===0); }
  else { ml.textContent='Empty board'; mg.textContent='+0'; mg.classList.add('zero'); }

  document.getElementById('loopcount').textContent = loops.length?`(${loops.length})`:'';
  const ll=document.getElementById('looplist');
  if(!loops.length){ ll.innerHTML='<div class="empty">none yet</div>'; }
  else { ll.innerHTML = loops.map(lp=>{
    const col=`var(${lp._c})`; const who = lp.owner===0?'A':'B';
    return `<div class="looprow"><span class="sw" style="background:${col}"></span>`+
           `<span style="color:var(--muted)">${who}</span> area <b class="mono" style="margin-left:2px">${lp.area}</b>`+
           `<span class="len mono">${lp.length} arcs</span></div>`;
  }).join(''); }

  document.getElementById('slider').value=s;
  document.getElementById('stepnum').textContent=s;
  lastRendered=s;
}

// controls
let playing=false, timer=null;
function setPlay(p){ playing=p; document.getElementById('play').textContent = p?'❚❚ Pause':'▶ Play';
  if(p){ if(state>=37) render(0,false); tick(); } else if(timer){ clearTimeout(timer); } }
function tick(){ if(!playing) return; if(state>=37){ setPlay(false); return; }
  render(state+1,true);
  timer=setTimeout(tick, parseInt(document.getElementById('speed').value)); }
document.getElementById('play').onclick=()=>setPlay(!playing);
document.getElementById('next').onclick=()=>{ setPlay(false); if(state<37) render(state+1,true); };
document.getElementById('prev').onclick=()=>{ setPlay(false); if(state>0) render(state-1,false); };
document.getElementById('slider').oninput=e=>{ setPlay(false); render(parseInt(e.target.value),false); };
addEventListener('keydown',e=>{ if(e.key==='ArrowRight'){document.getElementById('next').click();}
  else if(e.key==='ArrowLeft'){document.getElementById('prev').click();}
  else if(e.key===' '){e.preventDefault();setPlay(!playing);} });

render(0,false);
</script>"""

html = TEMPLATE.replace("__GAME_DATA__", json.dumps(data, separators=(",", ":")))
with open(os.path.join(here, "viewer.html"), "w") as f:
    f.write(html)
print("wrote viewer.html", len(html), "bytes")
