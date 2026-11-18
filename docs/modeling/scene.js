import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

// ── 명세 #2-6 최소 조건 ────────────────────────────────────────────
const W = 25, H = 6, D = 17;            // 공장 넓이 25m × 17m, 높이 6m
const FEAT = ["ni_pct","co_pct","li_pct","mn_pct","cu_pct","fe_pct","al_pct",
              "feed_kg","acid_l","temp_c","time_hr","arrhenius"];
const COMP = ["ni_pct","co_pct","li_pct","mn_pct","cu_pct","fe_pct","al_pct"];
const GRADES = ["A","B","C"];
const GRADE_DESC = { A:"고니켈 · 저불순물", B:"중간", C:"저니켈 · 고불순물" };
const GRADE_COL  = { A:0x2a2f38, B:0x3b362e, C:0x4b3027 };

const $ = id => document.getElementById(id);
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const clamp = (v,a,b) => Math.min(b, Math.max(a, v));
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

let D_ = null;                           // data.json
const state = { batch:null, acid:900, temp:70, time:4, cam:"전경" };

// ── 모델 (파이썬 Ridge 를 그대로 옮긴 것) ──────────────────────────
function featRow(b, acid, temp, time) {
  const v = Object.assign({}, b, { acid_l:acid, temp_c:temp, time_hr:time });
  v.arrhenius = Math.exp(-3000 / (temp + 273.15));
  return FEAT.map(k => v[k]);
}
function ridge(spec, x) {
  let s = spec.intercept;
  for (let i = 0; i < x.length; i++) s += ((x[i] - spec.mean[i]) / spec.scale[i]) * spec.coef[i];
  return s;
}
function domainDist(acid, temp, time) {
  const m = D_.domain, q = [acid, temp, time].map((v,i) => (v - m.mean[i]) / m.scale[i]);
  const ds = m.train.map(p => Math.hypot(
    (p[0]-m.mean[0])/m.scale[0] - q[0],
    (p[1]-m.mean[1])/m.scale[1] - q[1],
    (p[2]-m.mean[2])/m.scale[2] - q[2]));
  ds.sort((a,b) => a-b);
  return ds.slice(0, m.k).reduce((a,b) => a+b, 0) / m.k;
}
// 목표 달성 가부는 모듈 3(파이썬)이 계산한 추천 조건에서만 판정한다.
// 브라우저는 불확실성 σ 를 정확히 계산할 수 없어서(트리 앙상블), 다른 조건에서
// 확률 제약을 다시 계산하면 파이썬과 판정이 어긋난다 — 표본 54건 중 5건에서 확인.
// 회수율 평균 · 산 비율 · 적용 범위 거리는 파이썬과 같은 값이므로 그대로 계산한다.
let OVERRIDE = null;                     // Streamlit 이 넘긴 추천 (목표를 바꿨을 수 있다)
function recFor(b) {
  if (OVERRIDE && OVERRIDE.batch_id === b.batch_id) return OVERRIDE;
  return D_.rec?.[b.batch_id];
}
function evaluate() {
  const b = state.batch, { acid, temp, time } = state;
  const x = featRow(b, acid, temp, time);
  const ni = ridge(D_.ridge.ni_recovery_pct, x);
  const co = ridge(D_.ridge.co_recovery_pct, x);
  const ratio = acid / b.feed_kg;
  const dist  = domainDist(acid, temp, time);
  const inRatio  = ratio >= D_.ratio.min && ratio <= D_.ratio.max;
  const inDomain = dist <= D_.domain.threshold;
  const rec = recFor(b);
  const atRec = !!rec && rec.a != null && Math.abs(rec.a - acid) < 1e-6
    && Math.abs(rec.t - temp) < 1e-6 && Math.abs(rec.h - time) < 1e-6;
  const verdict = atRec ? (rec.ok ? "ok" : "short")
    : !inRatio ? "ratio" : !inDomain ? "domain" : "explore";
  return { ni, co, ratio, dist, rec, atRec, verdict, ok: verdict === "ok" };
}

// ── 캔버스 텍스처 유틸 ────────────────────────────────────────────
function canvasTex(w, h, draw) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function floorTex() {
  return canvasTex(1536, 1044, (g, w, h) => {
    g.fillStyle = "#8d918d"; g.fillRect(0, 0, w, h);
    // 콘크리트 패널 줄눈 (1m 격자)
    g.strokeStyle = "#7d817d"; g.lineWidth = 3;
    for (let i = 0; i <= 25; i++) { const x = i/25*w; g.beginPath(); g.moveTo(x,0); g.lineTo(x,h); g.stroke(); }
    for (let i = 0; i <= 17; i++) { const y = i/17*h; g.beginPath(); g.moveTo(0,y); g.lineTo(w,y); g.stroke(); }
    // 침출 구역 해치 (반응조 둘레)
    g.save();
    g.translate(w*0.62, h*0.5);
    g.strokeStyle = "#c8a233"; g.lineWidth = 10;
    g.strokeRect(-150, -150, 300, 300);
    g.lineWidth = 7;
    for (let i = -150; i < 150; i += 26) {
      g.beginPath(); g.moveTo(i, -150); g.lineTo(i+22, -128); g.stroke();
      g.beginPath(); g.moveTo(i, 150); g.lineTo(i+22, 128); g.stroke();
    }
    g.restore();
    // 통로 중앙선
    g.strokeStyle = "#b9bfb4"; g.lineWidth = 14; g.setLineDash([60, 38]);
    g.beginPath(); g.moveTo(0, h*0.84); g.lineTo(w, h*0.84); g.stroke();
    g.setLineDash([]);
    g.fillStyle = "#6f736f"; g.font = "600 46px ui-monospace, monospace";
    g.fillText("LEACHING  AREA  /  침출 구역", w*0.40, h*0.22);
    g.font = "600 34px ui-monospace, monospace";
    g.fillText("25.0 m × 17.0 m", 40, h - 36);
  });
}
function beltTex() {
  const t = canvasTex(256, 64, (g, w, h) => {
    g.fillStyle = "#24262a"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "#3b3f45"; g.lineWidth = 8;
    for (let i = -h; i < w + h; i += 34) {
      g.beginPath(); g.moveTo(i, h); g.lineTo(i + h*0.6, 0); g.stroke();
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(10, 1);
  return t;
}
function plateTex(lines, sub) {
  return canvasTex(512, 256, (g, w, h) => {
    g.fillStyle = "#1d2124"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "#5d666b"; g.lineWidth = 6; g.strokeRect(10, 10, w-20, h-20);
    g.fillStyle = "#e8ece9"; g.textAlign = "center";
    g.font = "700 72px ui-monospace, monospace";
    g.fillText(lines, w/2, h/2 - 4);
    if (sub) { g.font = "500 36px sans-serif"; g.fillStyle = "#a9b2ae"; g.fillText(sub, w/2, h/2 + 54); }
  });
}

// ── 씬 ───────────────────────────────────────────────────────────
const canvas = $("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 16/10, 0.1, 400);

// 궤도 조작 — 드래그로 회전, 휠/핀치로 확대. 외부 addon 을 쓰지 않는다
// (addon 이 bare specifier 'three' 를 import 해 importmap 을 요구하기 때문).
class Orbit {
  constructor(cam, el) {
    this.cam = cam;
    this.target = new THREE.Vector3();
    this.sph = new THREE.Spherical(33, 1.0, 0.8);
    this.min = 6; this.max = 76; this.maxPolar = Math.PI * 0.492;
    this.vel = { t:0, p:0 };
    this.ptrs = new Map();
    this.pinch = 0;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", e => {
      this.ptrs.set(e.pointerId, { x:e.clientX, y:e.clientY });
      try { el.setPointerCapture(e.pointerId); } catch {}
    });
    el.addEventListener("pointermove", e => {
      const prev = this.ptrs.get(e.pointerId);
      if (!prev) return;
      this.ptrs.set(e.pointerId, { x:e.clientX, y:e.clientY });
      if (this.ptrs.size >= 2) {
        const [a, b] = [...this.ptrs.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (this.pinch) this.zoom(1 - (d - this.pinch) / 220);
        this.pinch = d;
        return;
      }
      this.vel.t -= (e.clientX - prev.x) * 0.0052;
      this.vel.p -= (e.clientY - prev.y) * 0.0052;
    });
    const drop = e => { this.ptrs.delete(e.pointerId); if (this.ptrs.size < 2) this.pinch = 0; };
    el.addEventListener("pointerup", drop);
    el.addEventListener("pointercancel", drop);
    el.addEventListener("pointerleave", drop);
    el.addEventListener("wheel", e => {
      e.preventDefault();
      this.zoom(1 + Math.sign(e.deltaY) * 0.085);
    }, { passive:false });
  }
  zoom(f) { this.sph.radius = clamp(this.sph.radius * f, this.min, this.max); }
  setFrom(pos, target) {
    this.target.copy(target);
    this.sph.setFromVector3(pos.clone().sub(target));
    this.sph.radius = clamp(this.sph.radius, this.min, this.max);
    this.sph.phi = clamp(this.sph.phi, 0.06, this.maxPolar);
    this.vel.t = this.vel.p = 0;
  }
  update() {
    this.sph.theta += this.vel.t;
    this.sph.phi = clamp(this.sph.phi + this.vel.p, 0.06, this.maxPolar);
    this.vel.t *= 0.87; this.vel.p *= 0.87;
    this.cam.position.copy(this.target).add(new THREE.Vector3().setFromSpherical(this.sph));
    this.cam.lookAt(this.target);
  }
}
const controls = new Orbit(camera, canvas);

const MAT = {
  concrete: new THREE.MeshStandardMaterial({ color:0x8e928e, roughness:0.95 }),
  wall:     new THREE.MeshStandardMaterial({ color:0xa8aca6, roughness:0.9, side:THREE.BackSide }),
  steel:    new THREE.MeshStandardMaterial({ color:0x9aa2a6, roughness:0.38, metalness:0.75 }),
  dark:     new THREE.MeshStandardMaterial({ color:0x3c4348, roughness:0.55, metalness:0.5 }),
  beam:     new THREE.MeshStandardMaterial({ color:0x6d7479, roughness:0.7, metalness:0.4 }),
  pipe:     new THREE.MeshStandardMaterial({ color:0x7f8a86, roughness:0.45, metalness:0.6 }),
  acidPipe: new THREE.MeshStandardMaterial({ color:0xb08a2a, roughness:0.5, metalness:0.4 }),
  rail:     new THREE.MeshStandardMaterial({ color:0xc8a233, roughness:0.6, metalness:0.3 }),
  belt:     new THREE.MeshStandardMaterial({ map:beltTex(), roughness:0.85 }),
  lamp:     new THREE.MeshStandardMaterial({ color:0xfff3d6, emissive:0xfff0cc, emissiveIntensity:1.1 }),
};

// 조명
scene.add(new THREE.HemisphereLight(0xdfe6e4, 0x4a4f4c, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(13, 16, 11);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -20; sun.shadow.camera.right = 20;
sun.shadow.camera.top = 20;   sun.shadow.camera.bottom = -20;
sun.shadow.camera.far = 60;
scene.add(sun);
scene.add(new THREE.DirectionalLight(0xbfd2d8, 0.35).translateY(6));

// ── 맵 1개: 공장 셸 ───────────────────────────────────────────────
const map = new THREE.Group();
map.name = "factory_inner";
map.userData.spec = "map";

const room = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), MAT.wall);
room.position.y = H/2;
room.name = "shell";
map.add(room);

const slab = new THREE.Mesh(new THREE.BoxGeometry(W, 0.24, D),
  new THREE.MeshStandardMaterial({ map:floorTex(), roughness:0.95 }));
slab.position.y = -0.12;
slab.receiveShadow = true;
map.add(slab);

// 천장 트러스 6연
for (let i = 0; i < 6; i++) {
  const z = -D/2 + 1.4 + i * (D - 2.8) / 5;
  const web = new THREE.Mesh(new THREE.BoxGeometry(W - 0.4, 0.34, 0.14), MAT.beam);
  web.position.set(0, H - 0.5, z); map.add(web);
  const top = new THREE.Mesh(new THREE.BoxGeometry(W - 0.4, 0.1, 0.5), MAT.beam);
  top.position.set(0, H - 0.3, z); map.add(top);
  for (let j = 0; j < 9; j++) {
    const br = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.72, 0.1), MAT.beam);
    br.position.set(-W/2 + 1.6 + j * (W - 3.2) / 8, H - 0.86, z);
    map.add(br);
  }
}
// 벽면 파이프 랙
for (const [y, r] of [[4.9, 0.17], [4.5, 0.12], [4.15, 0.1]]) {
  const p = new THREE.Mesh(new THREE.CylinderGeometry(r, r, W - 0.6, 14), MAT.pipe);
  p.rotation.z = Math.PI/2;
  p.position.set(0, y, -D/2 + 0.55);
  map.add(p);
}
for (let i = 0; i < 5; i++) {
  const br = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 1.0), MAT.beam);
  br.position.set(-9 + i * 4.5, 4.52, -D/2 + 0.55);
  map.add(br);
}
// 조명 기구 3기 (PointLight 는 3개로 제한)
for (const x of [-8, 1, 9]) {
  const housing = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.18, 0.5), MAT.dark);
  housing.position.set(x, H - 1.05, 1.5);
  map.add(housing);
  const lens = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.06, 0.36), MAT.lamp);
  lens.position.set(x, H - 1.16, 1.5);
  map.add(lens);
  const pl = new THREE.PointLight(0xfff0d4, 24, 16, 2);
  pl.position.set(x, H - 1.4, 1.5);
  map.add(pl);
}
scene.add(map);

// ── 설비 1개: 침출 반응조 R-101 ───────────────────────────────────
const RX = 3.4, RZ = 0;
const reactor = new THREE.Group();
reactor.name = "leaching_reactor_R101";
reactor.userData.spec = "facility";
reactor.position.set(RX, 0, RZ);

const addTo = (g, m, cast = true) => { m.castShadow = cast; m.receiveShadow = true; g.add(m); return m; };

addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(2.25, 2.25, 0.5, 32), MAT.dark)
  .translateY(0.25));
addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(2.2, 0.5, 1.0, 32), MAT.steel)
  .translateY(1.0));
const body = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 3.1, 32), MAT.steel)
  .translateY(3.05));
addTo(reactor, new THREE.Mesh(new THREE.SphereGeometry(2.2, 32, 12, 0, Math.PI*2, 0, Math.PI/2), MAT.steel)
  .translateY(4.6));
// 보강 밴드
for (const y of [2.0, 3.1, 4.2]) {
  const band = addTo(reactor, new THREE.Mesh(new THREE.TorusGeometry(2.23, 0.06, 8, 40), MAT.dark), false);
  band.rotation.x = Math.PI/2;
  band.position.set(0, y, 0);
}
// 교반기 구동부
addTo(reactor, new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.85, 1.0), MAT.dark).translateY(5.45));
addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.3, 16), MAT.steel).translateY(4.92));

// 애니메이션 ①: 교반기
const agitator = new THREE.Group();
agitator.position.set(0, 0, 0);
addTo(agitator, new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 3.7, 12), MAT.steel)
  .translateY(3.05));
for (const y of [1.65, 2.95]) {
  for (let i = 0; i < 3; i++) {
    const bl = addTo(agitator, new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.05, 0.38), MAT.steel));
    bl.position.set(Math.cos(i*2.094)*0.78, y, Math.sin(i*2.094)*0.78);
    bl.rotation.y = i * 2.094;
    bl.rotation.x = 0.35;
  }
  addTo(agitator, new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.22, 12), MAT.dark)
    .translateY(y), false);
}
reactor.add(agitator);

// 액체 (애니메이션 ③ 이 쓴다)
const liquidMat = new THREE.MeshStandardMaterial({
  color:0x2f7f86, roughness:0.16, metalness:0.1, transparent:true, opacity:0.9 });
const liquid = new THREE.Mesh(new THREE.CylinderGeometry(2.14, 2.14, 1, 32), liquidMat);
liquid.name = "liquid";
reactor.add(liquid);

// 황산 주입 배관 + 밸브
const acidRun = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 5.6, 14), MAT.acidPipe), false);
acidRun.rotation.z = Math.PI/2;
acidRun.position.set(-4.4, 5.05, 0);
const acidDrop = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.9, 14), MAT.acidPipe), false);
acidDrop.position.set(-1.6, 4.7, 0);
const valve = addTo(reactor, new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.07, 8, 20), MAT.rail), false);
valve.rotation.y = Math.PI/2;
valve.position.set(-3.1, 5.05, 0);
// 배출 배관
const drainUp = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.9, 14), MAT.pipe), false);
drainUp.position.set(0, 0.2, 0);
const drainRun = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 5.5, 14), MAT.pipe), false);
drainRun.rotation.x = Math.PI/2;
drainRun.position.set(0, 0.22, 3.1);

// 점검 발판 + 난간 + 사다리
const deck = addTo(reactor, new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.12, 5.0), MAT.beam));
deck.position.set(3.3, 2.7, 0);
for (const z of [-2.5, 2.5]) {
  addTo(reactor, new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.7, 0.12), MAT.beam), false)
    .position.set(4.5, 1.35, z);
  addTo(reactor, new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.7, 0.12), MAT.beam), false)
    .position.set(2.2, 1.35, z);
}
for (const y of [3.2, 3.68]) {
  const bar = addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 5.0, 8), MAT.rail), false);
  bar.rotation.x = Math.PI/2;
  bar.position.set(4.55, y, 0);
}
for (const z of [-2.4, -1.2, 0, 1.2, 2.4]) {
  addTo(reactor, new THREE.Mesh(new THREE.BoxGeometry(0.07, 1.05, 0.07), MAT.rail), false)
    .position.set(4.55, 3.25, z);
}
for (let i = 0; i < 7; i++) {
  addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.66, 8), MAT.rail), false)
    .rotateZ(Math.PI/2).position.set(5.2, 0.35 + i * 0.36, 2.0);
}
for (const z of [1.67, 2.33]) {
  addTo(reactor, new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 2.8, 8), MAT.rail), false)
    .position.set(5.2, 1.4, z);
}
// 명판
const plate = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.85),
  new THREE.MeshBasicMaterial({ map:plateTex("R-101", "침출 반응조") }));
plate.position.set(0, 3.5, 2.21);
reactor.add(plate);
scene.add(reactor);

// ── 광석 3개: 등급별 블랙파우더 호퍼 ──────────────────────────────
const ores = {};
GRADES.forEach((g, i) => {
  const grp = new THREE.Group();
  grp.name = `ore_${g}`;
  grp.userData.spec = "ore";
  grp.userData.grade = g;
  grp.position.set(-9.2 + i * 3.1, 0, -4.6);

  addTo(grp, new THREE.Mesh(new THREE.CylinderGeometry(1.15, 0.3, 1.5, 24), MAT.steel)
    .translateY(2.0));
  addTo(grp, new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.15, 0.7, 24), MAT.steel)
    .translateY(3.1));
  for (let j = 0; j < 4; j++) {
    addTo(grp, new THREE.Mesh(new THREE.BoxGeometry(0.11, 2.6, 0.11), MAT.beam), false)
      .position.set(Math.cos(j*1.571)*0.95, 1.3, Math.sin(j*1.571)*0.95);
  }
  // 블랙파우더 더미 — 등급마다 색이 다르다
  const powder = new THREE.MeshStandardMaterial({ color:GRADE_COL[g], roughness:0.98 });
  const pile = addTo(grp, new THREE.Mesh(new THREE.ConeGeometry(1.05, 0.8, 22), powder));
  pile.position.y = 3.6;
  grp.userData.pile = pile;
  // 바닥 더미도 하나
  const heap = addTo(grp, new THREE.Mesh(new THREE.SphereGeometry(0.85, 18, 10), powder));
  heap.position.set(0, 0.06, 1.9); heap.scale.set(1, 0.35, 1);

  const lab = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.75),
    new THREE.MeshBasicMaterial({ map:plateTex(`${g}급`, GRADE_DESC[g]) }));
  lab.position.set(0, 3.15, 1.16);
  grp.add(lab);

  // 선택 표시 링
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1.35, 0.05, 8, 32),
    new THREE.MeshBasicMaterial({ color:0xc8a233 }));
  ring.rotation.x = Math.PI/2; ring.position.y = 0.04; ring.visible = false;
  grp.add(ring);
  grp.userData.ring = ring;

  ores[g] = grp;
  scene.add(grp);
});

// ── 컨베이어 (애니메이션 ②) ───────────────────────────────────────
const conveyor = new THREE.Group();
const C0 = new THREE.Vector3(-3.4, 1.5, -4.6);     // 호퍼 쪽 시작
const C1 = new THREE.Vector3(RX - 0.6, 5.5, RZ);   // 반응조 투입구
const span = C1.clone().sub(C0);
const len = span.length();
const belt = new THREE.Mesh(new THREE.BoxGeometry(len, 0.08, 1.0), MAT.belt);
belt.position.copy(C0).add(span.clone().multiplyScalar(0.5));
belt.lookAt(C1);
belt.rotateY(Math.PI/2);
belt.castShadow = true;
conveyor.add(belt);
// 프레임과 롤러
for (let i = 0; i <= 8; i++) {
  const p = C0.clone().add(span.clone().multiplyScalar(i/8));
  const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 1.05, 10), MAT.dark);
  roll.position.set(p.x, p.y - 0.09, p.z);
  roll.rotation.x = Math.PI/2;
  roll.rotation.y = Math.atan2(span.x, span.z) + Math.PI/2;
  conveyor.add(roll);
  if (i % 2 === 0 && p.y > 0.6) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.1, p.y - 0.2, 0.1), MAT.beam);
    leg.position.set(p.x, (p.y - 0.2)/2, p.z);
    leg.castShadow = true;
    conveyor.add(leg);
  }
}
// 이송 중인 원료 입자
const PARTS = 26;
const partGeo = new THREE.DodecahedronGeometry(0.1);
const partMat = new THREE.MeshStandardMaterial({ color:GRADE_COL.B, roughness:0.98 });
const parts = new THREE.InstancedMesh(partGeo, partMat, PARTS);
parts.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
const partT = new Array(PARTS).fill(0).map((_, i) => i / PARTS);
const partJit = new Array(PARTS).fill(0).map(() => (Math.random() - 0.5) * 0.6);
conveyor.add(parts);
scene.add(conveyor);

// ── 치수 실측 + 명세 검증 ─────────────────────────────────────────
const bbox = new THREE.Box3().setFromObject(room);
const dim = { w:bbox.max.x - bbox.min.x, h:bbox.max.y - bbox.min.y, d:bbox.max.z - bbox.min.z };
const ANIMS = [
  { id:"agitator", nm:"교반기 회전", ds:"R-101 임펠러 2단", on:!REDUCED },
  { id:"conveyor", nm:"원료 이송·투입", ds:"컨베이어 + 낙하", on:!REDUCED },
  { id:"acidfill", nm:"황산 주입·승온", ds:"액위 + 온도 색", on:!REDUCED },
];
function countSpec(role) {
  let n = 0;
  scene.traverse(o => { if (o.userData.spec === role) n++; });
  return n;
}
function renderSpec() {
  const nMap = countSpec("map"), nFac = countSpec("facility"), nOre = countSpec("ore");
  const sizeOk = dim.w >= 25 && dim.d >= 17 && dim.h >= 6;
  const nOk = D_.batches.filter(b => D_.rec?.[b.batch_id]?.ok).length;
  const rows = [
    ["공장 실측", `${dim.w.toFixed(1)} × ${dim.d.toFixed(1)} × ${dim.h.toFixed(1)} m`, sizeOk],
    ["맵", `${nMap} / 1`, nMap >= 1],
    ["설비", `${nFac} / 1`, nFac >= 1],
    ["애니메이션", `${ANIMS.length} / 3`, ANIMS.length >= 3],
    ["광석", `${nOre} / 3`, nOre >= 3],
  ];
  $("spec").innerHTML = rows.map(([k, v, p]) =>
    `<span class="chip" data-pass="${p ? 1 : 0}"><span class="k">${k}</span><span class="v">${v}</span></span>`
  ).join("")
  + `<span class="chip"><span class="k">명세</span><span class="v">#2-6 최소 조건</span></span>`
  + `<span class="chip"><span class="k">목표 달성 가능한 원료</span><span class="v">${nOk} / ${D_.batches.length}</span></span>`;
}

// ── 카메라 프리셋 ─────────────────────────────────────────────────
const CAMS = {
  "전경":   { p:[22, 13, 21], t:[1, 2.2, 0] },
  "반응조": { p:[10.5, 5.2, 7.5], t:[RX, 3.0, RZ] },
  "투입부": { p:[-4.5, 4.4, 3.6], t:[-6.2, 2.4, -4.0] },
  "평면도": { p:[0.5, 31, 0.6], t:[0, 0, 0] },
};
function setCam(name, snap) {
  const c = CAMS[name]; if (!c) return;
  state.cam = name;
  const to = new THREE.Vector3(...c.p), tt = new THREE.Vector3(...c.t);
  if (snap) { controls.setFrom(to, tt); camTween = null; }
  else { camTween = { from:camera.position.clone(), to, tfrom:controls.target.clone(), tto:tt, k:0 }; }
  [...$("cams").children].forEach(b => b.setAttribute("aria-pressed", String(b.textContent === name)));
}
let camTween = null;
$("cams").innerHTML = Object.keys(CAMS).map(n => `<button type="button">${n}</button>`).join("");
[...$("cams").children].forEach(b => b.addEventListener("click", () => setCam(b.textContent, REDUCED)));

// ── UI ───────────────────────────────────────────────────────────
function renderAnimToggles() {
  $("anim").innerHTML = ANIMS.map((a, i) => `<label class="tog" for="an${i}">
    <input type="checkbox" id="an${i}" ${a.on ? "checked" : ""}>
    <span><span class="nm">${i+1}. ${a.nm}</span><br><span class="ds">${a.ds}</span></span></label>`).join("");
  ANIMS.forEach((a, i) => $(`an${i}`).addEventListener("change", e => { a.on = e.target.checked; }));
}

function renderPanel() {
  const b = state.batch, r = evaluate();
  $("vacid").textContent = `${state.acid} L`;
  $("vtemp").textContent = `${state.temp} ℃`;
  $("vtime").textContent = `${state.time.toFixed(1)} hr`;

  const imp = b.cu_pct + b.fe_pct + b.al_pct;
  $("feed").innerHTML = [
    ["등급", `${b.grade}급`], ["Ni / Co", `${b.ni_pct} / ${b.co_pct} %`],
    ["불순물 합", `${imp.toFixed(1)} %`], ["투입량", `${b.feed_kg} kg`],
    ["산 비율", `${r.ratio.toFixed(2)} L/kg`], ["판정 거리", `${r.dist.toFixed(3)} / ${D_.domain.threshold.toFixed(3)}`],
  ].map(([k, v]) => `<div>${k} <b>${v}</b></div>`).join("");

  const tg = { ni: r.rec?.ni_target ?? D_.targets.ni, co: r.rec?.co_target ?? D_.targets.co };
  const V = {
    ok:      { pill: "운전 가능", cls: "ok", title: "모듈 3 추천 조건 — 목표 달성",
               msg: `Ni ${tg.ni}% · Co ${tg.co}% 목표를 확률 제약(예측 평균 − ${D_.targets.z}σ ≥ 목표)으로 만족한다. `
                  + "금속별 판정이며, σ는 보정된 확률이 아니다." },
    short:   { pill: "목표 미달", cls: "bad", title: "모듈 3 판정 — 목표 미달",
               msg: `허용 범위 안에서 가장 근접한 조건이다. 확률 제약(예측 평균 − ${D_.targets.z}σ) 기준으로 목표에 못 미친다`
                  + (r.rec?.gap != null ? ` (부족분 ${r.rec.gap}%p).` : ".") + " 불순물이 낮은 원료와 블렌딩을 검토한다." },
    ratio:   { pill: "거절", cls: "bad", title: "비율 제약 위반",
               msg: `산 비율 ${r.ratio.toFixed(2)} L/kg 는 관측 범위 ${D_.ratio.min}–${D_.ratio.max} 밖이다.` },
    domain:  { pill: "거절", cls: "bad", title: "적용 범위 밖",
               msg: `학습점까지 평균 거리 ${r.dist.toFixed(3)} 가 임계 ${D_.domain.threshold.toFixed(3)} 를 넘는다. 추천하지 않는다.` },
    explore: { pill: "참고", cls: "ref", title: "참고 계산 — 가부 판정 안 함",
               msg: "회수율 평균은 학습된 모델과 같은 값이다. 목표 달성 여부는 불확실성 계산이 필요해 모듈 3 추천 조건에서만 판정한다." },
  }[r.verdict];
  let msg = V.msg;
  if (!r.atRec && r.rec) {
    msg += r.rec.ok
      ? ` 이 원료의 추천 조건은 ${r.rec.a} L · ${r.rec.t} ℃ · ${r.rec.h} hr 이다 — 아래 버튼이 그리로 옮겨 준다.`
      : ` 이 원료는 허용 범위 안에 달성 조건이 없다 (최소 부족 ${r.rec.gap ?? "?"}%p).`;
  }
  $("vpill").textContent = V.pill;
  $("vpill").className = "pill " + V.cls;
  $("vtitle").textContent = V.title;
  $("vmsg").textContent = msg;
  $("rd").innerHTML = [
    ["Ni 회수율 (예측 평균)", r.ni.toFixed(1), `목표 ${tg.ni}%`],
    ["Co 회수율 (예측 평균)", r.co.toFixed(1), `목표 ${tg.co}%`],
  ].map(([k, v, d]) => `<div class="b"><div class="k">${k}</div>
    <div class="v" style="color:${r.ok ? "var(--ok)" : "var(--fg)"}">${v}<span style="font-size:11px"> %</span></div>
    <div class="d">${d}</div></div>`).join("");

  $("hud").innerHTML = [
    ["교반", `${ANIMS[0].on ? 85 : 0} rpm`], ["액위", `${liqTarget().toFixed(2)} m`],
    ["액온", `${state.temp} ℃`], ["상태", V.pill],
  ].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");

  GRADES.forEach(g => { ores[g].userData.ring.visible = (g === b.grade); });
  partMat.color.setHex(GRADE_COL[b.grade]);
}

function liqTarget() {   // 황산 400–1500 L → 액위 0.5–2.9 m
  return 0.5 + (state.acid - 400) / 1100 * 2.4;
}

function applyRecipe(r) {
  state.acid = clamp(Math.round(r.a / 10) * 10, 400, 1500);
  state.temp = clamp(Math.round(r.t), 40, 95);
  state.time = clamp(Math.round(r.h * 2) / 2, 1, 8);
  $("acid").value = state.acid; $("temp").value = state.temp; $("time").value = state.time;
  renderPanel();
}

function selectBatch(b) {
  state.batch = b;
  const r = recFor(b);
  if (r) applyRecipe(r);                 // 원료를 고르면 그 원료의 추천 레시피를 올린다
  else renderPanel();
  const note = $("gonote");
  {
    note.textContent = r?.ok
      ? "추천값 적용됨. 다른 조건은 참고 계산만 표시한다."
      : `이 원료는 허용 범위 안에 달성 조건이 없다 (최소 부족 ${r?.gap ?? "?"}%p). 가장 근접한 조건을 올렸다.`;
  }
}

function wire() {
  const sel = $("batch");
  D_.batches.forEach(b => {
    const o = document.createElement("option");
    const ok = D_.rec?.[b.batch_id]?.ok;
    o.value = b.batch_id;
    o.textContent = `${ok ? "✓" : "✕"} ${b.batch_id}  ${b.grade}급  Ni ${b.ni_pct.toFixed(1)}%  불순물 ${(b.cu_pct+b.fe_pct+b.al_pct).toFixed(1)}%`;
    sel.appendChild(o);
  });
  // Streamlit 임베드는 파이썬이 계산한 진짜 추천을 window.__TWIN_OPENING__ 으로 넘긴다.
  // 정적 페이지에서는 비어 있으므로 data.json 의 미리 계산값을 쓴다.
  const ov = (typeof window !== "undefined" && window.__TWIN_OPENING__) || null;
  if (ov && D_.batches.some(b => b.batch_id === ov.batch_id)) OVERRIDE = ov;
  const op = OVERRIDE || D_.opening;
  const start = (op && D_.batches.find(b => b.batch_id === op.batch_id)) || D_.batches[0];
  sel.value = start.batch_id;
  state.batch = start;
  const r0 = (op && op.a != null ? op : null)
    || recFor(start)
    || { a:start.acid_l || 900, t:start.temp_c || 70, h:start.time_hr || 4 };
  state.acid = clamp(Math.round(r0.a / 10) * 10, 400, 1500);
  state.temp = clamp(Math.round(r0.t), 40, 95);
  state.time = clamp(Math.round(r0.h * 2) / 2, 1, 8);
  $("acid").value = state.acid; $("temp").value = state.temp; $("time").value = state.time;

  sel.addEventListener("change", e => selectBatch(D_.batches.find(b => b.batch_id === e.target.value)));
  $("acid").addEventListener("input", e => { state.acid = +e.target.value; renderPanel(); });
  $("temp").addEventListener("input", e => { state.temp = +e.target.value; renderPanel(); });
  $("time").addEventListener("input", e => { state.time = +e.target.value; renderPanel(); });

  $("go").addEventListener("click", () => {
    const r = recFor(state.batch);
    if (!r || r.a == null) return;
    applyRecipe(r);
    $("gonote").textContent = r.ok
      ? `모듈 3 추천 — ${r.a} L · ${r.t} ℃ · ${r.h} hr (허용 범위 안 비용 최소해)`
      : `허용 범위 안에 달성 조건이 없다. 부족분 ${r.gap ?? "?"}%p 인 최근접 조건을 올렸다.`;
    setCam("반응조", REDUCED);
  });
}

// ── 테마 ─────────────────────────────────────────────────────────
function applyTheme() {
  const bg = new THREE.Color(cssv("--scene") || "#b9c0bd");
  scene.background = bg;
  scene.fog = new THREE.Fog(bg, 46, 128);
  MAT.wall.color.set(cssv("--scene2") || "#9ba3a0");
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, { attributeFilter:["data-theme"] });

// ── 리사이즈 ─────────────────────────────────────────────────────
function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

// ── 루프 ─────────────────────────────────────────────────────────
const dummy = new THREE.Object3D();
let liqNow = 1.0;
const clock = new THREE.Clock();

function tick() {
  const dt = Math.min(clock.getDelta(), 0.05);

  if (ANIMS[0].on) agitator.rotation.y += 8.9 * dt;          // ① 85 rpm

  if (ANIMS[1].on) {                                          // ② 이송
    MAT.belt.map.offset.x -= dt * 0.55;
    for (let i = 0; i < PARTS; i++) {
      partT[i] = (partT[i] + dt * 0.14) % 1.18;
      const t = partT[i];
      if (t <= 1) {
        const p = C0.clone().add(span.clone().multiplyScalar(t));
        dummy.position.set(p.x, p.y + 0.11, p.z + partJit[i]);
      } else {                                                 // 투입구에서 낙하
        const f = (t - 1) / 0.18;
        dummy.position.set(C1.x + (RX - C1.x) * f, C1.y + 0.11 - f * f * 2.6, C1.z + partJit[i] * 0.3);
      }
      dummy.rotation.set(t * 9, t * 7, t * 5);
      dummy.updateMatrix();
      parts.setMatrixAt(i, dummy.matrix);
    }
    parts.instanceMatrix.needsUpdate = true;
  }
  parts.visible = ANIMS[1].on;

  const tgt = liqTarget();                                     // ③ 액위 · 색
  liqNow += (tgt - liqNow) * (ANIMS[2].on ? Math.min(1, dt * 2.6) : 1);
  const bob = ANIMS[2].on ? Math.sin(clock.elapsedTime * 1.7) * 0.012 : 0;
  liquid.scale.y = Math.max(0.02, liqNow + bob);
  liquid.position.y = 1.5 + liquid.scale.y / 2 - 0.5;
  const heat = clamp((state.temp - 40) / 55, 0, 1);
  liquidMat.color.setRGB(0.12 + heat * 0.72, 0.46 - heat * 0.18, 0.49 - heat * 0.35);

  if (camTween) {
    camTween.k = Math.min(1, camTween.k + dt * 1.6);
    const e = 1 - Math.pow(1 - camTween.k, 3);
    camera.position.lerpVectors(camTween.from, camTween.to, e);
    controls.target.lerpVectors(camTween.tfrom, camTween.tto, e);
    camera.lookAt(controls.target);
    if (camTween.k >= 1) { controls.setFrom(camera.position, controls.target); camTween = null; }
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

// ── 시작 ─────────────────────────────────────────────────────────
fetch("./data.json").then(r => r.json()).then(d => {
  D_ = d;
  applyTheme();
  wire();
  renderAnimToggles();
  renderSpec();
  renderPanel();
  setCam("전경", true);
  resize();
  $("load").hidden = true;
  tick();
}).catch(err => { $("load").textContent = "데이터를 불러오지 못했습니다: " + err.message; });
