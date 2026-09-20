const container = document.getElementById('scene-container');
if (!container) throw new Error("No scene-container found");

// --- 2. ESCENA Y RENDERIZADOR ---
const canvas = document.getElementById('canvas3d');
const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: false, powerPreference: "high-performance" });
renderer.setClearColor(0x0a0a0a);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Limitar densidad de píxeles
renderer.autoClear = false;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

let splitView = false;
window.setSplitView = (val) => { splitView = val; resize(); };

const camera1 = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera1.position.set(5, 8, 15); // Frente a la mosca (que está en x=5)
const controls1 = new THREE.OrbitControls(camera1, renderer.domElement);
controls1.target.set(5, 2, 0); // Mirando al centro de la mosca

const camera2 = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
camera2.position.set(2, 6, 8);
const controls2 = new THREE.OrbitControls(camera2, renderer.domElement);
controls2.target.set(5, 2, 0); // Mirando a la cabeza de la mosca

// Vistas reproducibles para inspección anatómica y capturas.
window.setCameraView = (view) => {
    const views = {
        side: [17.0, 3.0, 0.0],
        dorsal: [5, 13.0, 0.8],
        front: [5, 3.2, 12.0]
    };
    const position = views[view] || views.side;
    camera1.position.set(...position); controls1.target.set(5, 1.35, 0); controls1.update();
};

// Lógica de activación de controles por hover
let mouseX = 0;
container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
});

const ambientLight = new THREE.AmbientLight(0xdcecff, 1.8);
scene.add(ambientLight);
const hemiLight = new THREE.HemisphereLight(0xc8e7ff, 0x31404c, 1.7);
scene.add(hemiLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 2.4);
dirLight.position.set(4, 12, 10);
scene.add(dirLight);
const rimLight = new THREE.PointLight(0x69d8ff, 3.5, 18);
rimLight.position.set(5, 5, 6);
scene.add(rimLight);

// --- 3. MODELO MOSCA PROCEDURAL AVANZADO ---
const flyGroup = new THREE.Group();
scene.add(flyGroup);

// Cáscara Fresnel barata: permite ver el cerebro y el cordón dentro del cuerpo.
const shellMat = new THREE.ShaderMaterial({
    uniforms: { shellColor: { value: new THREE.Color(0x385c61) }, shellOpacity: { value: 0.16 } },
    vertexShader: `varying vec3 vNormal; varying vec3 vViewPosition;
        void main(){ vec4 p=modelViewMatrix*vec4(position,1.0); vNormal=normalize(normalMatrix*normal);
        vViewPosition=-p.xyz; gl_Position=projectionMatrix*p; }`,
    fragmentShader: `uniform vec3 shellColor; uniform float shellOpacity; varying vec3 vNormal; varying vec3 vViewPosition;
        void main(){ float f=pow(1.0-abs(dot(normalize(vNormal),normalize(vViewPosition))),2.2);
        gl_FragColor=vec4(shellColor*(0.3+1.5*f),shellOpacity*(0.5+f)); }`,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
});
const cuticleMat = new THREE.MeshStandardMaterial({ color: 0x4a3324, roughness: 0.72 });
const abdomenCanvas = document.createElement('canvas'); abdomenCanvas.width = 64; abdomenCanvas.height = 256;
const abdomenCtx = abdomenCanvas.getContext('2d');
for (let row = 0; row < abdomenCanvas.height; row++) {
    const tergite = Math.floor(row / 34);
    abdomenCtx.fillStyle = tergite % 2 === 0 ? '#4a2b1b' : '#916438';
    abdomenCtx.fillRect(0, row, abdomenCanvas.width, 1);
    if (row % 34 < 4) { abdomenCtx.fillStyle = '#17100d'; abdomenCtx.fillRect(0, row, abdomenCanvas.width, 4); }
}
const abdomenTex = new THREE.CanvasTexture(abdomenCanvas);
const abdomenMat = new THREE.MeshStandardMaterial({ map: abdomenTex, color: 0xffffff, roughness: 0.78 });
const eyeMat = new THREE.MeshStandardMaterial({ color: 0x5b1020, emissive: 0x1e0208, roughness: 0.32 });
const legMat = new THREE.MeshStandardMaterial({ color: 0x221711, roughness: 0.78 });

// Cabeza y tórax en cáscara; ajustado para anatomía real (la cabeza se une al tórax de frente, no por encima)
const head = new THREE.Mesh(new THREE.SphereGeometry(1.0, 32, 20), shellMat);
head.position.set(0, 1.45, 1.95); // Más baja y pegada al tórax
head.scale.set(1.15, 0.9, 0.85); // Más ancha
flyGroup.add(head);

// Ojos compuestos (enormes, cubriendo los laterales de la cabeza)
const eyeGeo = new THREE.IcosahedronGeometry(0.85, 2);
for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeo, eyeMat);
    eye.position.set(side * 0.75, 1.48, 2.05); // Pegados a los lados
    eye.scale.set(0.6, 1.1, 0.85); // Ovalados verticalmente
    eye.rotation.z = side * 0.2;
    flyGroup.add(eye);
}
const ocellusGeo = new THREE.SphereGeometry(0.11, 10, 8);
for (const [x, z] of [[-0.18, 1.72], [0, 1.82], [0.18, 1.72]]) {
    const ocellus = new THREE.Mesh(ocellusGeo, eyeMat); 
    ocellus.position.set(x, 2.25, z); // Encima de la cabeza ajustada
    flyGroup.add(ocellus);
}

// Antenas (pequeñas, apuntando hacia abajo desde el frente)
const antMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
for (const side of [-1, 1]) {
    const antenna = new THREE.Group(); 
    antenna.position.set(side * 0.25, 1.1, 2.7);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.08, 0.6, 6), antMat);
    shaft.position.set(side * 0.05, -0.2, 0.1); 
    shaft.rotation.z = side * 0.2; 
    shaft.rotation.x = 0.5; // Apuntando hacia abajo
    antenna.add(shaft);
    
    const arista = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.4, 4), antMat);
    arista.position.set(side * 0.1, -0.4, 0.3);
    arista.rotation.x = 1.0;
    antenna.add(arista);
    flyGroup.add(antenna);
}

const thorax = new THREE.Mesh(new THREE.SphereGeometry(1.35, 32, 24), shellMat);
thorax.position.set(0, 1.4, 0.4); // Más horizontal
thorax.scale.set(0.9, 0.95, 1.35); // Más alargado hacia atrás
flyGroup.add(thorax);

// Abdomen horizontal: LatheGeometry
const abdomenProfile = [[0.10,-2.35],[0.30,-2.18],[0.48,-1.82],[0.60,-1.35],[0.68,-0.75],[0.70,-0.10],[0.63,0.55],[0.48,1.15],[0.25,1.55],[0.08,1.72]];
const abdomen = new THREE.Mesh(new THREE.LatheGeometry(abdomenProfile.map(([r,y]) => new THREE.Vector2(r,y)), 32), abdomenMat);
abdomen.position.set(0, 1.25, -1.95); // Ligeramente caído
abdomen.rotation.x = -Math.PI / 2 + 0.3; // Inclinado hacia abajo
abdomen.scale.setScalar(1.05);
flyGroup.add(abdomen);


// Halterios con pequeña maza.
for (const side of [-1, 1]) {
    const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 0.42, 8), antMat);
    stalk.position.set(side * 0.9, 1.45, -0.75); stalk.rotation.z = side * 1.2; flyGroup.add(stalk);
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 8), eyeMat);
    knob.position.set(side * 1.15, 1.25, -0.75); knob.scale.set(1, 0.7, 1); flyGroup.add(knob);
}

// Alas elípticas con venación procedural, plegadas hacia atrás horizontalmente
const wingCanvas = document.createElement('canvas'); wingCanvas.width = 256; wingCanvas.height = 128;
const wingCtx = wingCanvas.getContext('2d'); wingCtx.strokeStyle = 'rgba(150,200,220,0.8)'; wingCtx.lineWidth = 2;
wingCtx.beginPath(); wingCtx.ellipse(128, 64, 116, 48, -0.16, 0, Math.PI * 2); wingCtx.stroke();
for (let i = -4; i <= 4; i++) { wingCtx.beginPath(); wingCtx.moveTo(22, 64); wingCtx.quadraticCurveTo(125, 64 + i * 10, 238, 64 + i * 7); wingCtx.stroke(); }
const wingTex = new THREE.CanvasTexture(wingCanvas);
const wingMat = new THREE.MeshBasicMaterial({ map: wingTex, color: 0xddffff, transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide });
const wingMeshes = [];
for (const side of [-1, 1]) {
    const wing = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 1.5), wingMat);
    wing.position.set(side * 0.6, 2.3, -1.2); 
    wing.rotation.set(1.5, side * 0.1, side * 0.15); // Plegadas planas sobre el abdomen (X ~ 1.5 rad)
    wing.scale.x = side; 
    flyGroup.add(wing);
    wingMeshes.push(wing);
}

// Patas: coxa ventral, fémur grueso, tibia fina y tarso de cinco segmentos.
const legs = [];
function makeBone(length, radius) {
    return new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.72, radius, length, 8), legMat);
}
function pointBone(mesh, from, to) {
    const delta = new THREE.Vector3().subVectors(to, from);
    mesh.position.copy(from).add(to).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
}
function createLeg(side, z, direction) {
    const root = new THREE.Group(); flyGroup.add(root);
    const leg = {
        root, side, z, direction, hip: new THREE.Vector3(side * 0.82, 0.84, z),
        lengths: { coxa: 0.42, femur: 0.92, tibia: 1.05 },
        coxa: makeBone(0.42, 0.13), femur: makeBone(0.92, 0.11), tibia: makeBone(1.05, 0.065),
        tarsus: [], foot: new THREE.Vector3()
    };
    root.add(leg.coxa, leg.femur, leg.tibia);
    for (let i = 0; i < 5; i++) { leg.tarsus.push(makeBone(0.13, 0.032)); root.add(leg.tarsus[i]); }
    leg.footHome = new THREE.Vector3(side * 1.65, 0.04, z + direction * 1.55);
    return leg;
}
legs.push(createLeg(-1, 1.1, 1), createLeg(1, 1.1, 1));
legs.push(createLeg(-1, 0.2, 0), createLeg(1, 0.2, 0));
legs.push(createLeg(-1, -0.75, -1), createLeg(1, -0.75, -1));

function solveLeg(leg, targetLocal) {
    const hip = leg.hip.clone();
    const coxaEnd = hip.clone().add(new THREE.Vector3(leg.side * 0.34, -0.18, leg.direction * 0.16));
    pointBone(leg.coxa, hip, coxaEnd);
    const l1 = leg.lengths.femur, l2 = leg.lengths.tibia;
    const delta = new THREE.Vector3().subVectors(targetLocal, coxaEnd);
    const distance = THREE.MathUtils.clamp(delta.length(), Math.abs(l1 - l2) + 0.02, l1 + l2 - 0.02);
    const dir = delta.normalize();
    const bend = new THREE.Vector3(0, -1, leg.direction * 0.28).cross(dir).normalize();
    if (!Number.isFinite(bend.x)) bend.set(0, 0, 1);
    const along = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance);
    const height = Math.sqrt(Math.max(0, l1 * l1 - along * along));
    const knee = coxaEnd.clone().addScaledVector(dir, along).addScaledVector(bend, height);
    const ankle = targetLocal.clone();
    pointBone(leg.femur, coxaEnd, knee); pointBone(leg.tibia, knee, ankle);
    const step = ankle.clone().sub(knee).normalize();
    for (let i = 0; i < leg.tarsus.length; i++) {
        const a = ankle.clone().addScaledVector(step, i * 0.13);
        pointBone(leg.tarsus[i], a, a.clone().addScaledVector(step, 0.13));
    }
    leg.foot.copy(ankle);
}
function restTripod() {
    const tripodA = [0, 3, 4], tripodB = [1, 2, 5];
    for (const index of tripodA) solveLeg(legs[index], legs[index].footHome.clone().setY(0.04 + Math.sin(Date.now() * 0.003) * 0.025));
    for (const index of tripodB) solveLeg(legs[index], legs[index].footHome.clone().setY(0.04 - Math.sin(Date.now() * 0.003) * 0.025));
}
function positionFlyForTarget(targetWorld, leg) {
    flyGroup.position.x = targetWorld.x + 1.6;
    flyGroup.position.z = targetWorld.z - leg.hip.z;
    flyGroup.updateMatrixWorld(true);
}
window.runLegReachabilityTest = () => {
    const saved = flyGroup.position.clone(); let maxError = 0;
    for (let rank = 0; rank < 8; rank++) for (let file = 0; file < 8; file++) {
        const target = new THREE.Vector3(file - 3.5, 0.04, rank - 3.5);
        const leg = target.z > 0.8 ? legs[0] : target.z < -0.8 ? legs[4] : legs[2];
        positionFlyForTarget(target, leg); solveLeg(leg, flyGroup.worldToLocal(target.clone()));
        const footWorld = leg.foot.clone().applyMatrix4(flyGroup.matrixWorld);
        maxError = Math.max(maxError, footWorld.distanceTo(target));
    }
    flyGroup.position.copy(saved); restTripod();
    return { maxError, passed: maxError < 0.15 };
};

flyGroup.position.set(5, 0, 0); // Mosca a la derecha del tablero

// --- 4. RED NEURONAL REAL (LineSegments) ---
const numNeurons = 2000;
const positions = new Float32Array(numNeurons * 3);
const colors = new Float32Array(numNeurons * 3);

for(let i=0; i<numNeurons; i++) {
    if(i < numNeurons * 0.50) {
        // Cerebro central: verde.
        positions[i*3] = (Math.random() - 0.5) * 1.5;
        positions[i*3+1] = 2.0 + (Math.random() - 0.5) * 1.2;
        positions[i*3+2] = 2.5 + (Math.random() - 0.5) * 0.8;
        colors[i*3] = 0.27; colors[i*3+1] = 0.88; colors[i*3+2] = 0.55;
    } else if (i < numNeurons * 0.75) {
        // Lóbulos ópticos: morado, a ambos lados de la cabeza.
        positions[i*3] = (i % 2 ? 1 : -1) * (0.65 + Math.random() * 0.65);
        positions[i*3+1] = 2.0 + (Math.random() - 0.5) * 1.0;
        positions[i*3+2] = 2.4 + (Math.random() - 0.5) * 0.7;
        colors[i*3] = 0.72; colors[i*3+1] = 0.28; colors[i*3+2] = 1.0;
    } else {
        // Cordón nervioso ventral: azul y alargado hacia abdomen.
        positions[i*3] = (Math.random() - 0.5) * 0.5;
        positions[i*3+1] = 0.7 + (Math.random() - 0.5) * 0.45;
        positions[i*3+2] = -1.0 + (Math.random() - 0.5) * 3.2;
        colors[i*3] = 0.25; colors[i*3+1] = 0.67; colors[i*3+2] = 1.0;
    }
}
const brainPositionAttr = new THREE.BufferAttribute(positions, 3);
const brainColorAttr = new THREE.BufferAttribute(colors, 3);
const brainPointsGeo = new THREE.BufferGeometry();
brainPointsGeo.setAttribute('position', brainPositionAttr); brainPointsGeo.setAttribute('color', brainColorAttr);

// Dibujamos como puntos con additive blending
const brainMat = new THREE.PointsMaterial({ 
    size: 0.055, sizeAttenuation: true, vertexColors: true, 
    transparent: true, opacity: 0.52, depthWrite: false, 
    blending: THREE.NormalBlending
});
brainMat.depthTest = false;
const brainPoints = new THREE.Points(brainPointsGeo, brainMat);
brainPoints.renderOrder = 20;
flyGroup.add(brainPoints);

// Trazar algunas líneas entre neuronas para simular axones/conectoma (ELIMINADO POR RENDIMIENTO GRÁFICO)

// --- 5. TABLERO 3D ---
const boardGroup = new THREE.Group();
const squareSize = 1;
for(let row=0; row<8; row++) {
    for(let col=0; col<8; col++) {
        const isWhite = (row + col) % 2 !== 0;
        const mat = new THREE.MeshStandardMaterial({ color: isWhite ? 0xeeeed2 : 0x769656, roughness: 0.5 });
        const geo = new THREE.BoxGeometry(squareSize, 0.2, squareSize);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(col - 3.5, 0, row - 3.5);
        boardGroup.add(mesh);
    }
}
scene.add(boardGroup);

// Piezas 3D simples sincronizadas con el FEN: suficiente para seguir el movimiento.
const pieces3dGroup = new THREE.Group(); scene.add(pieces3dGroup);
let moveHighlights3d = [];
function makePiece3D(piece) {
    const white = piece === piece.toUpperCase();
    const material = new THREE.MeshStandardMaterial({ color: white ? 0xf0ead8 : 0x241b1a, roughness: 0.55 });
    const group = new THREE.Group();
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 0.12, 16), material); base.position.y = 0.16; group.add(base);
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.24, 0.42, 12), material); body.position.y = 0.42; group.add(body);
    if ('qk'.includes(piece.toLowerCase())) {
        const top = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), material); top.position.y = 0.72; group.add(top);
    } else if ('nr'.includes(piece.toLowerCase())) {
        const top = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.32, 8), material); top.position.y = 0.72; group.add(top);
    } else {
        const top = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), material); top.position.y = 0.7; group.add(top);
    }
    return group;
}
function updatePieces3D(fen) {
    pieces3dGroup.clear();
    const rows = fen.split(' ')[0].split('/');
    for (let row = 0; row < 8; row++) {
        let file = 0;
        for (const char of rows[row]) {
            if (!isNaN(char)) { file += parseInt(char); continue; }
            const piece = makePiece3D(char); piece.position.set(file - 3.5, 0, row - 3.5);
            pieces3dGroup.add(piece); file++;
        }
    }
}
function highlight3D(from, to) {
    moveHighlights3d.forEach(mesh => boardGroup.remove(mesh)); moveHighlights3d = [];
    for (const square of [from, to]) if (square >= 0) {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.035, 0.86), new THREE.MeshBasicMaterial({ color: 0xffd740, transparent: true, opacity: 0.58 }));
        mesh.position.set(square % 8 - 3.5, 0.13, Math.floor(square / 8) - 3.5); boardGroup.add(mesh); moveHighlights3d.push(mesh);
    }
}

// --- 6. RENDER Y ANIMACIÓN ---
function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    if(camera1) {
        camera1.aspect = (splitView ? width/2 : width) / height;
        camera1.updateProjectionMatrix();
    }
    if(camera2) {
        camera2.aspect = (splitView ? width/2 : width) / height;
        camera2.updateProjectionMatrix();
    }
}
window.addEventListener('resize', resize);
resize();

let targetPoint = null;
let ikActive = false;

function animateIK(leg, target) {
    if (!target) return;
    positionFlyForTarget(target, leg);
    solveLeg(leg, flyGroup.worldToLocal(target.clone()));
}

function animate() {
    requestAnimationFrame(animate);
    const t = Date.now() * 0.005;
    
    // Marcha en trípode: tres patas levantan unos milímetros alternadamente.
    if (!ikActive) {
        flyGroup.position.lerp(new THREE.Vector3(5, 0, 0), 0.08);
        restTripod();
    }
    
    wingMeshes[0].rotation.y = -0.44 + Math.sin(t * 10) * 0.05;
    wingMeshes[1].rotation.y = 0.44 - Math.sin(t * 10) * 0.05;
    
    if (ikActive && targetPoint) {
        animateIK(legs[0], targetPoint); // Mueve la pata delantera izquierda al objetivo
    }
    
    const w = container.clientWidth;
    const h = container.clientHeight;
    
    // Lógica Hover para Controles
    if (splitView) {
        controls1.enabled = mouseX < w/2;
        controls2.enabled = mouseX >= w/2;
        controls1.update();
        controls2.update();
        
        renderer.setScissorTest(true);
        renderer.setViewport(0, 0, w/2, h);
        renderer.setScissor(0, 0, w/2, h);
        renderer.render(scene, camera1);
        
        renderer.setViewport(w/2, 0, w/2, h);
        renderer.setScissor(w/2, 0, w/2, h);
        renderer.render(scene, camera2);
    } else {
        controls1.enabled = true;
        controls2.enabled = false;
        controls1.update();
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, w, h);
        renderer.render(scene, camera1);
    }
}
animate();

// --- 7. UI Y TABLERO 2D ---
const boardEl = document.getElementById('board');
const candidatesList = document.getElementById('candidates-list');
const sfMove = document.getElementById('sf-move');

const pieceMap = {
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔'
};

function renderBoard2D(fen) {
    boardEl.innerHTML = '';
    const rows = fen.split(' ')[0].split('/');
    for(let r=0; r<8; r++) {
        let col = 0;
        for(let i=0; i<rows[r].length; i++) {
            const char = rows[r][i];
            if(isNaN(char)) {
                // Es pieza
                const isWhiteSq = (r + col) % 2 !== 0;
                const sq = document.createElement('div');
                sq.className = 'square ' + (isWhiteSq ? 'white-sq' : 'black-sq');
                sq.innerText = pieceMap[char];
                boardEl.appendChild(sq);
                col++;
            } else {
                // Es espacio vacío
                const spaces = parseInt(char);
                for(let s=0; s<spaces; s++) {
                    const isWhiteSq = (r + col) % 2 !== 0;
                    const sq = document.createElement('div');
                    sq.className = 'square ' + (isWhiteSq ? 'white-sq' : 'black-sq');
                    boardEl.appendChild(sq);
                    col++;
                }
            }
        }
    }
    updatePieces3D(fen);
}
renderBoard2D('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');

// --- 8. WEBSOCKET LOGIC ---
const ws = new WebSocket(`ws://${location.host}/ws/replay`);
const phaseEl = document.getElementById('phase');
const neuralCountEl = document.getElementById('neural-count');
const neuralSelectionEl = document.getElementById('neural-selection');
const rasterEl = document.getElementById('raster');
const thresholdEl = document.getElementById('spike-threshold');
let neuralMeta = [];
let spikeThreshold = Number(thresholdEl?.value || 0.96);
thresholdEl?.addEventListener('input', event => { spikeThreshold = Number(event.target.value); });
function renderRaster(spikes) {
    if (!rasterEl) return;
    rasterEl.innerHTML = '';
    spikes.slice(0, 80).forEach((id, index) => {
        const spike = document.createElement('span'); spike.className = 'raster-spike';
        spike.style.left = `${(index / 80) * 100}%`; spike.title = `neurona ${id}`; rasterEl.appendChild(spike);
    });
    if (neuralCountEl) neuralCountEl.textContent = `${spikes.length} spikes`;
}

ws.onmessage = function(event) {
    if (event.data.includes("error")) return;
    const data = JSON.parse(event.data);
    phaseEl.textContent = data.phase || "RECIBIENDO POSICIÓN";
    if (data.neural_meta) neuralMeta = data.neural_meta;
    if (data.spikes) {
        renderRaster(data.spikes);
        const selected = neuralMeta[data.spikes[0] % Math.max(1, neuralMeta.length)];
        if (selected && neuralSelectionEl) neuralSelectionEl.textContent = `Neuron ${selected.id} · ${selected.region} · ${selected.type}`;
    }
    
    // UI Updates
    if (data.fen) renderBoard2D(data.fen);
    highlight3D(data.action_from, data.action_to);
    
    if (data.candidates) {
        candidatesList.innerHTML = '';
        data.candidates.forEach((c, idx) => {
            const row = document.createElement('div');
            row.className = 'candidate-row ' + (idx === 0 ? 'chosen' : '');
            row.innerHTML = `<span>${c.move}</span><span>${(c.prob * 100).toFixed(1)}%</span>`;
            candidatesList.appendChild(row);
        });
    }
    
    // Cerebro 3D Updates
    if (data.rates) {
        const cols = brainColorAttr.array;
        for(let i=0; i<numNeurons; i++) {
            const r = data.rates[i]; // 1:1 mapping directly from backend
            if (r > 0) {
                cols[i*3] = 0.28 + Math.random()*0.22; cols[i*3+1] = 0.62; cols[i*3+2] = 0.18;
            } else {
                cols[i*3] *= 0.8; cols[i*3+1] *= 0.8; cols[i*3+2] *= 0.8;
                if(cols[i*3] < 0.1) { cols[i*3]=0.1; cols[i*3+1]=0.2; cols[i*3+2]=0.3; }
            }
        }
        brainColorAttr.needsUpdate = true;
    }
    // Propagación demo por sinapsis: ilumina también el destino de cada spike.
    if (data.synapses) for (const [, destination] of data.synapses) {
        const i = destination % numNeurons; brainColorAttr.array[i*3] = 0.45; brainColorAttr.array[i*3+1] = 0.85; brainColorAttr.array[i*3+2] = 1.0;
    }
    
    // Movimiento basado en neuronas descendentes
    if (data.desc_rates > 0.5 && data.action_from >= 0) {
        ikActive = true;
        const sq = data.desc_rates > 0.8 ? data.action_to : data.action_from; // Umbral de "decisión"
        if (sq >= 0) {
            const r = Math.floor(sq / 8);
            const c = sq % 8;
            targetPoint = new THREE.Vector3(c - 3.5, 0, r - 3.5);
        }
    } else {
        ikActive = false;
    }
};
