// [audio.js] 우리 숲의 모든 '소리'를 만들어내는 마법 같은 파일입니다.
// 드럼, 실로폰 소리부터 숲속의 바람소리, 새소리까지 모두 여기서 관리하고 연주합니다.

let audioCtx = null; // 모든 소리를 만들어내기 위한 '마법 엔진' 같은 것입니다.
let masterGain = null; // 전체 볼륨을 조절하는 조절기입니다.
let ambientGain = null; // 배경음 볼륨을 조절하는 조절기입니다.
let dryGain = null; // 원래 소리 볼륨입니다.
let wetGain = null; // 울림 효과가 섞인 소리 볼륨입니다.
let convolver = null; // 숲속에 있는 것처럼 소리가 울리게(잔향) 만드는 장치입니다.
let limiter = null; // 너무 큰 소리가 나서 귀가 아프지 않게 소리 크기를 제한하는 장치입니다.
let delayNode = null; // 소리가 메아리처럼 늦게 들리게 하는 효과입니다.
let delayFeedback = null; // 메아리가 얼마나 많이 반복될지 결정합니다.
let delayMix = null; // 메아지 소리를 얼마나 섞을지 결정합니다.

let ambientTimer = null; // 바람 소리 주기를 조절하는 시계입니다.
let windNode = null; // 바람 소리를 발생시키는 장치입니다.
let windGain = null; // 바람 소리 크기를 조절합니다.
let birdTimer = null; // 새소리 발생 주기를 관리합니다.
let dropTimer = null; // 나뭇잎 소리 발생 주기를 관리합니다.

const SAMPLE_BASE = "/assets/sounds"; // 소리 파일들이 저장된 기본 폴더 주소입니다.
const SAMPLE_URLS = { // 각 악기별 실제 소리 파일 위치들입니다.
  drum: [`${SAMPLE_BASE}/드럼.mp3`, `${SAMPLE_BASE}/drum.mp3`, `${SAMPLE_BASE}/drum-mushroom.mp3`], // 드럼 소리입니다.
  xylophone: [`${SAMPLE_BASE}/피리.mp3`, `${SAMPLE_BASE}/whistle.mp3`, `${SAMPLE_BASE}/xylophone.mp3`, `${SAMPLE_BASE}/xylophone-vine.mp3`], // 피리(백합) 소리 우선입니다.
  tambourine: [`${SAMPLE_BASE}/피아노.mp3`, `${SAMPLE_BASE}/piano.mp3`, `${SAMPLE_BASE}/tambourine.mp3`, `${SAMPLE_BASE}/tambourine-flower.mp3`], // 피아노(클로버) 소리 우선입니다.
  fist: `${SAMPLE_BASE}/fist-beat.mp3`, // 주먹질 소리입니다.
  pinky: [`${SAMPLE_BASE}/심벌즈.mp3`, `${SAMPLE_BASE}/cymbals.mp3`, `${SAMPLE_BASE}/fern.mp3`, `${SAMPLE_BASE}/pinky-chime.mp3`], // 심벌즈(고사리) 소리 우선입니다.
  heart: [`${SAMPLE_BASE}/cat.mp3`, `${SAMPLE_BASE}/owl.mp3`, `${SAMPLE_BASE}/heart-bloom.mp3`], // 고양이(랜덤 동물) 소리 우선입니다.
  animalCat: [`${SAMPLE_BASE}/고양이.mp3`, `${SAMPLE_BASE}/cat.mp3`], // 애니멀 랜덤용 고양이 소리입니다.
  animalDog: [`${SAMPLE_BASE}/강아지.mp3`, `${SAMPLE_BASE}/dog.mp3`], // 애니멀 랜덤용 강아지 소리입니다.
  animal: `${SAMPLE_BASE}/animal-roll.mp3` // 랜덤 파일이 없을 때 폴백 효과음입니다.
};

const sampleBuffers = new Map();
const sampleLoadPromises = new Map();
const customSampleBuffers = new Map();
const customSampleLoadPromises = new Map();
let playbackContext = null;
let transportAnchorSec = null;

const INSTRUMENT_GRID_SEC = 0.3;
const TRANSPORT_LOOKAHEAD_SEC = 0.02;
const MIN_RETRIGGER_SEC = 0.12;

const instrumentScheduleState = {
  drum: { lastTime: -Infinity, nextGrid: 0 },
  tambourine: { lastTime: -Infinity, nextGrid: 0 },
  xylophone: { lastTime: -Infinity, nextGrid: 0 },
  whistle: { lastTime: -Infinity, nextGrid: 0 },
  triangle: { lastTime: -Infinity, nextGrid: 0 },
  animal: { lastTime: -Infinity, nextGrid: 0 }
};

export function setPlaybackContext(context) {
  playbackContext = context && typeof context === "object" ? { ...context } : null;
}

function emitSoundPlayed(soundKey, playMode) {
  const now = performance.now();
  const ctx = playbackContext;
  const detail = {
    soundKey,
    playMode,
    audioTs: now,
    instrumentId: ctx?.instrumentId || null,
    gestureLabel: ctx?.gestureLabel || null,
    gestureSource: ctx?.gestureSource || null,
    triggerTs: Number.isFinite(ctx?.triggerTs) ? ctx.triggerTs : null,
    latencyMs: Number.isFinite(ctx?.triggerTs) ? now - ctx.triggerTs : null
  };

  window.dispatchEvent(new CustomEvent("jamjam:sound-played", { detail }));
  playbackContext = null;
}

function nowTime() { // 현재 소리 엔진의 정확한 시각을 알려줍니다.
  return audioCtx ? audioCtx.currentTime : 0; // 엔진이 켜져 있으면 시간을, 아니면 0을 돌려줍니다.
}

function reserveInstrumentStart(instrumentType) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return nowTime();

  if (transportAnchorSec === null) {
    transportAnchorSec = ctx.currentTime;
  }

  if (!instrumentScheduleState[instrumentType]) {
    instrumentScheduleState[instrumentType] = { lastTime: -Infinity, nextGrid: 0 };
  }
  const state = instrumentScheduleState[instrumentType];
  const now = ctx.currentTime + TRANSPORT_LOOKAHEAD_SEC;

  if (now - state.lastTime < MIN_RETRIGGER_SEC) {
    return state.lastTime;
  }

  const relative = Math.max(0, now - transportAnchorSec);
  const grid = Math.ceil(relative / INSTRUMENT_GRID_SEC);
  const scheduled = transportAnchorSec + (grid * INSTRUMENT_GRID_SEC);

  state.lastTime = scheduled;
  state.nextGrid = grid + 1;

  return scheduled;
}

function clearScheduledTimers() { // 예약된 소리 일정들을 모두 취소하는 기능입니다.
  if (ambientTimer) { // 바람 소리 시계가 있으면
    clearInterval(ambientTimer); // 멈춥니다.
    ambientTimer = null; // 정보를 비웁니다.
  }
  if (birdTimer) { // 새소리 시계가 있으면
    clearTimeout(birdTimer); // 멈춥니다.
    birdTimer = null; // 정보를 비웁니다.
  }
  if (dropTimer) { // 나뭇잎 소리 시계가 있으면
    clearTimeout(dropTimer); // 멈춥니다.
    dropTimer = null; // 정보를 비웁니다.
  }
}

function stopWindNode() { // 바람 소리를 완전히 멈추고 연결을 끊는 기능입니다.
  if (windNode) { // 바람 소리 장치(노드)가 있으면
    try {
      windNode.stop(); // 재생을 멈춥니다.
    } catch {
      // no-op
    }
    windNode.disconnect(); // 연결을 끊어서 메모리를 아낍니다.
    windNode = null; // 정보를 지웁니다.
  }
  if (windGain) { // 바람 볼륨 조절기가 있으면
    windGain.disconnect(); // 연결을 끊습니다.
    windGain = null; // 정보를 지웁니다.
  }
}

// 숲속의 울림(잔향) 효과를 만들기 위해 아주 짧은 가짜 소리를 생성하는 기능입니다.
function createForestReverbIR(ctx, duration = 2.1, decay = 2.6) {
  const length = Math.floor(ctx.sampleRate * duration); // 소리 데이터의 전체 길이를 계산합니다.
  const buffer = ctx.createBuffer(2, length, ctx.sampleRate); // 2채널(스테레오) 도화지를 만듭니다.

  for (let ch = 0; ch < 2; ch += 1) { // 왼쪽, 오른쪽 채널 각각에 대해 작업합니다.
    const data = buffer.getChannelData(ch); // 해당 채널의 빈 공간을 가져옵니다.
    for (let i = 0; i < length; i += 1) { // 전체 길이를 하나씩 채워나갑니다.
      const t = i / length; // 소리의 진행률(0~1)을 계산합니다.
      const envelope = Math.pow(1 - t, decay); // 시간이 지날수록 소리가 줄어드는 '포장지' 모양을 만듭니다.
      data[i] = (Math.random() * 2 - 1) * envelope; // 무작위 소음(치직)을 포장지 모양대로 입힙니다.
    }
  }

  return buffer; // 만들어진 울림 소리 데이터를 돌려줍니다.
}

// '치익~' 하는 백색 소음을 만드는 기능입니다. 바람 소리 등을 만들 때 사용합니다.
function createNoiseBuffer(ctx, seconds = 0.2, level = 1) {
  const size = Math.max(1, Math.floor(ctx.sampleRate * seconds)); // 필요한 소리 데이터 크기를 잽니다.
  const buffer = ctx.createBuffer(1, size, ctx.sampleRate); // 1채널(모노) 도화지를 만듭니다.
  const data = buffer.getChannelData(0); // 데이터를 적을 공간을 가져옵니다.
  for (let i = 0; i < size; i += 1) { // 크기만큼 반복하며
    data[i] = (Math.random() * 2 - 1) * level; // 무작위 숫자를 채워 넣어 소음을 만듭니다.
  }
  return buffer; // 소음 데이터를 돌려줍니다.
}

// 인터넷 서버에서 소리 파일을 다운로드해서 컴퓨터가 연주할 수 있게 준비하는 기능입니다.
async function loadSampleBuffer(sampleId) {
  const ctx = ensureAudioContext(); // 소리 엔진이 준비되어 있는지 확인합니다.
  if (!ctx) return null; // 엔진이 없으면 아무것도 못 합니다.

  if (sampleBuffers.has(sampleId)) { // 이미 불러온 소리라면
    return sampleBuffers.get(sampleId) || null; // 바로 그 소리를 돌려줍니다.
  }

  if (sampleLoadPromises.has(sampleId)) { // 현재 불러오는 중인 소리라면
    return sampleLoadPromises.get(sampleId) || null; // 불러오기가 끝날 때까지 기다리는 약속을 돌려줍니다.
  }

  const urls = (() => { // 이 소리를 찾을 수 있는 인터넷 주소 목록을 만듭니다.
    const entry = SAMPLE_URLS[sampleId];
    if (!entry) return [];
    return Array.isArray(entry) ? entry : [entry];
  })();
  if (urls.length === 0) return null; // 주소가 없으면 종료합니다.

  const loader = (async () => { // 실제로 데이터를 가져오는 내부 과정입니다.
    try {
      for (const url of urls) { // 여러 후보 주소를 하나씩 시도합니다.
        try {
          const res = await fetch(url); // 인터넷에서 파일을 가져옵니다.
          if (!res.ok) continue; // 가져오기 실패하면 다음 주소를 시도합니다.
          const ab = await res.arrayBuffer(); // 파일 데이터를 이진 데이터로 바꿉니다.
          const buf = await ctx.decodeAudioData(ab.slice(0)); // 이진 데이터를 소리 엔진용 데이터로 변환합니다.
          sampleBuffers.set(sampleId, buf); // 성공한 소리를 저장소에 담아둡니다.
          return buf; // 변환된 소리를 돌려줍니다.
        } catch {
          // try next candidate URL // 오류가 나면 다음 주소로 넘어갑니다.
        }
      }

      return null; // 모든 주소가 실패하면 없다고 합니다.
    } finally {
      sampleLoadPromises.delete(sampleId); // 작업이 끝났으니 대기 목록에서 지웁니다.
    }
  })();

  sampleLoadPromises.set(sampleId, loader); // 현재 작업 중임을 기록해둡니다.
  return loader; // 작업을 시작하고 약속을 돌려줍니다.
}

async function loadCustomSampleBuffer(customKey, dataUrl) {
  const ctx = ensureAudioContext();
  if (!ctx || !dataUrl) return null;

  if (customSampleBuffers.has(customKey)) {
    return customSampleBuffers.get(customKey) || null;
  }

  if (customSampleLoadPromises.has(customKey)) {
    return customSampleLoadPromises.get(customKey) || null;
  }

  const loader = (async () => {
    try {
      const res = await fetch(dataUrl);
      if (!res.ok) return null;
      const ab = await res.arrayBuffer();
      const buf = await ctx.decodeAudioData(ab.slice(0));
      customSampleBuffers.set(customKey, buf);
      return buf;
    } catch {
      return null;
    } finally {
      customSampleLoadPromises.delete(customKey);
    }
  })();

  customSampleLoadPromises.set(customKey, loader);
  return loader;
}

// 소리 파일들을 미리 읽어와서 연주할 준비를 해두는 기능입니다.
function preloadSamples() {
  Object.keys(SAMPLE_URLS).forEach((sampleId) => { // 등록된 모든 소리 이름을 하나씩 꺼냅니다.
    loadSampleBuffer(sampleId); // 각 소리를 미리 불러오기 시작합니다.
  });
}

export function preloadCustomSounds(customSoundMap = {}) {
  Object.entries(customSoundMap).forEach(([instrumentId, soundData]) => {
    if (!soundData?.data) return;
    loadCustomSampleBuffer(`custom_${instrumentId}`, soundData.data);
  });
}

// 준비된 소리 데이터를 가져와서 실제로 연주를 시작하는 기능입니다.
function playSample(sampleId, { wet = 0.22, delay = 0.12, gain = 1 } = {}) {
  const ctx = ensureAudioContext(); // 소리 엔진 상태를 확인합니다.
  if (!ctx || ctx.state !== "running") return false; // 엔진이 꺼져 있으면 연주하지 않습니다.

  const buffer = sampleBuffers.get(sampleId); // 저장된 소리 데이터를 가져옵니다.
  if (!buffer) { // 아직 데이터가 없으면
    loadSampleBuffer(sampleId); // 지금이라도 불러오기 시작합니다.
    return false; // 이번 연주는 패스합니다.
  }

  const src = ctx.createBufferSource(); // 새로운 재생기(턴테이블 같은 것)를 만듭니다.
  const out = ctx.createGain(); // 소리 크기 조절기를 만듭니다.
  out.gain.value = gain; // 설정된 볼륨으로 맞춥니다.

  src.buffer = buffer; // 재생기에 소리 데이터를 끼웁니다.
  src.connect(out); // 재생기를 조절기에 연결합니다.
  connectSource(out, wet, delay); // 조절기를 최종 출력 장치들에 연결합니다.
  src.start(); // 소리를 내기 시작합니다.
  emitSoundPlayed(sampleId, "sample");

  return true; // 연주 성공을 알립니다.
}

export function playCustomSample(customKey, dataUrl, { wet = 0.22, delay = 0.12, gain = 1 } = {}) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return false;
  if (!customKey || !dataUrl) return false;

  const buffer = customSampleBuffers.get(customKey);
  if (!buffer) {
    loadCustomSampleBuffer(customKey, dataUrl);
    return false;
  }

  const src = ctx.createBufferSource();
  const out = ctx.createGain();
  out.gain.value = gain;

  src.buffer = buffer;
  src.connect(out);
  connectSource(out, wet, delay);
  src.start();
  emitSoundPlayed(customKey, "custom-sample");

  return true;
}

function connectSource(node, wetAmount = 1, delayAmount = 0.22) { // 소리 노드를 여러 효과 필터에 연결하는 기능입니다.
  if (!dryGain || !wetGain || !delayMix) return; // 필수 조절기가 없으면 연결할 수 없습니다.
  node.connect(dryGain); // 생소리(Dry) 조절기에 연결합니다.

  if (wetAmount > 0) { // 울림(Wet) 효과를 넣으려면
    const wetSend = audioCtx.createGain(); // 울림으로 보낼 통로용 조절기를 만듭니다.
    wetSend.gain.value = wetAmount; // 얼마나 울리게 할지 정합니다.
    node.connect(wetSend); // 소리를 통로에 보냅니다.
    wetSend.connect(wetGain); // 통로를 울림 효과기(Convolver)에 연결합니다.
  }

  if (delayAmount > 0) { // 메아리(Delay) 효과를 넣으려면
    const delaySend = audioCtx.createGain(); // 메아리로 보낼 통로용 조절기를 만듭니다.
    delaySend.gain.value = delayAmount; // 얼마나 메아리치게 할지 정합니다.
    node.connect(delaySend); // 소리를 통로에 보냅니다.
    delaySend.connect(delayMix); // 통로를 메아리 효과기에 연결합니다.
  }
}

function connectAmbient(node) {
  if (!ambientGain) return;
  node.connect(ambientGain);
}

// 숲속에서 가끔 새가 지저귀는 소리가 들리도록 예약하는 기능입니다.
function scheduleBirdTone() {
  if (!audioCtx || audioCtx.state !== "running") return; // 엔진이 꺼져 있으면 새도 울지 않습니다.

  const waitMs = 2400 + Math.random() * 2600; // 2.4초에서 5초 사이의 무작위 시간을 정합니다.
  birdTimer = window.setTimeout(() => { // 정해진 시간 뒤에 새소리를 냅니다.
    if (!audioCtx || audioCtx.state !== "running") return; // 다시 한번 엔진 상태를 확인합니다.

    const t = nowTime(); // 현재 시각을 잽니다.
    const base = 980 + Math.random() * 220; // 새소리의 기본 높낮이(주파수)를 정합니다.
    const osc = audioCtx.createOscillator(); // 소리 파동 발생기(목청 역할)를 만듭니다.
    const gain = audioCtx.createGain(); // 볼륨 조절기를 만듭니다.
    const pan = audioCtx.createStereoPanner(); // 소리 위치(왼쪽/오른쪽) 조절기를 만듭니다.

    osc.type = "triangle"; // 부드러운 삼각형 모양 파동을 사용합니다.
    osc.frequency.setValueAtTime(base, t); // 처음 음높이를 설정합니다.
    osc.frequency.linearRampToValueAtTime(base + 180 + Math.random() * 120, t + 0.15); // 0.15초 동안 음높이를 빠르게 올립니다. (짹!)
    gain.gain.setValueAtTime(0.001, t); // 처음엔 소리가 안 나게 합니다.
    gain.gain.linearRampToValueAtTime(0.06, t + 0.03); // 0.03초 만에 소리를 키웁니다.
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35); // 0.35초 동안 소리를 서서히 줄여서 끝냅니다.
    pan.pan.value = Math.random() * 1.6 - 0.8; // 소리가 왼쪽이나 오른쪽 무작위 방향에서 들리게 합니다.

    osc.connect(gain); // 발생기를 조절기에 연결합니다.
    gain.connect(pan); // 조절기를 위치 조절기에 연결합니다.
    connectAmbient(pan); // 배경음 출력 장치에 연결합니다.

    osc.start(t); // 소리를 내기 시작합니다.
    osc.stop(t + 0.4); // 0.4초 뒤에 소리를 멈춥니다.

    scheduleBirdTone(); // 다음 새소리를 위해 다시 예약합니다. (무한 반복)
  }, waitMs);
}

// 나뭇잎이 떨어지는 듯한 아주 작은 소리를 무작위로 재생하는 기능입니다.
function scheduleLeafDrop() {
  if (!audioCtx || audioCtx.state !== "running") return; // 엔진이 꺼져 있으면 조용히 합니다.

  const waitMs = 1700 + Math.random() * 1800; // 1.7초에서 3.5초 사이의 무작위 대기 시간을 정합니다.
  dropTimer = window.setTimeout(() => { // 정해진 시간이 지나면 실행합니다.
    if (!audioCtx || audioCtx.state !== "running") return; // 엔진 상태를 재확인합니다.

    const t = nowTime(); // 현재 시각을 가져옵니다.
    const src = audioCtx.createBufferSource(); // 소음 재생기를 만듭니다.
    const hp = audioCtx.createBiquadFilter(); // 고주파 통과 필터(날카로운 소리만 남김)를 만듭니다.
    const bp = audioCtx.createBiquadFilter(); // 특정 대역 통과 필터(원하는 소리만 강조)를 만듭니다.
    const gain = audioCtx.createGain(); // 볼륨 조절기를 만듭니다.

    src.buffer = createNoiseBuffer(audioCtx, 0.08, 0.5); // 아주 짧은(0.08초) 소음을 만듭니다.
    hp.type = "highpass"; // 낮은 소리를 깎아냅니다.
    hp.frequency.value = 900; // 900Hz 이하의 웅웅거리는 소리를 지웁니다.
    bp.type = "bandpass"; // 중간의 바스락거리는 소리만 남깁니다.
    bp.frequency.value = 1800 + Math.random() * 900; // 바스락 소리의 높낮이를 무작위로 정합니다.
    bp.Q.value = 0.8; // 소리의 날카로운 정도를 정합니다.

    gain.gain.setValueAtTime(0.001, t); // 처음 소리입니다.
    gain.gain.exponentialRampToValueAtTime(0.03, t + 0.01); // 아주 빠르게 소리를 냅니다.
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09); // 0.09초 만에 소리를 사라지게 합니다.

    src.connect(hp); // 재생기를 첫 번째 필터에 연결합니다.
    hp.connect(bp); // 첫 필터를 두 번째 필터에 연결합니다.
    bp.connect(gain); // 두 번째 필터를 조절기에 연결합니다.
    connectAmbient(gain); // 배경음 출력에 연결합니다.

    src.start(t); // 소리를 재생합니다.
    src.stop(t + 0.1); // 0.1초 뒤에 멈춥니다.

    scheduleLeafDrop(); // 다음 바스락 소리를 예약합니다.
  }, waitMs);
}

export function ensureAudioContext() { // 소리 마법 엔진을 준비하고 각종 효과 장치들을 서로 연결하는 기능입니다.
  if (audioCtx) return audioCtx; // 이미 엔진이 있으면 그대로 사용합니다.

  const Ctx = window.AudioContext || window.webkitAudioContext; // 브라우저에서 제공하는 소리 엔진 틀을 가져옵니다.
  if (!Ctx) return null; // 소리 엔진을 지원하지 않는 브라우저면 그만둡니다.

  audioCtx = new Ctx(); // 새로운 소리 마법 엔진 인스턴스를 만듭니다.

  limiter = audioCtx.createDynamicsCompressor(); // 너무 큰 소리를 자동으로 줄여주는 '리미터'를 만듭니다.
  limiter.threshold.value = -14; // 어느 정도 크기부터 소리를 줄일지 정합니다.
  limiter.knee.value = 20; // 소리를 부드럽게 줄일지 정합니다.
  limiter.ratio.value = 4.5; // 소리를 얼마나 강하게 압축할지 정합니다.
  limiter.attack.value = 0.003; // 소리가 커졌을 때 얼마나 빨리 반응할지 정합니다.
  limiter.release.value = 0.22; // 소리가 다시 작아졌을 때 얼마나 천천히 풀지 정합니다.

  masterGain = audioCtx.createGain(); // 전체 볼륨을 조절하는 메인 볼륨 다이얼입니다.
  masterGain.gain.value = 1.48; // 기본 볼륨을 1.48배로 설정합니다.

  dryGain = audioCtx.createGain(); // 효과가 섞이지 않은 본래 소리의 볼륨 조절기입니다.
  dryGain.gain.value = 0.8; // 본래 소리는 80% 크기로 나옵니다.

  wetGain = audioCtx.createGain(); // 울림 효과가 적용될 소리의 볼륨 조절기입니다.
  wetGain.gain.value = 0.24; // 울림 소리는 24% 크기로 섞입니다.

  delayNode = audioCtx.createDelay(0.7); // 메아리 효과를 만드는 장치입니다. (최대 0.7초)
  delayNode.delayTime.value = 0.16; // 0.16초마다 메아리가 들리게 합니다.
  delayFeedback = audioCtx.createGain(); // 메아리가 다시 메아리치게 하는 피드백 조절기입니다.
  delayFeedback.gain.value = 0.28; // 메아리가 반복될수록 28%씩 소리가 줄어듭니다.
  delayMix = audioCtx.createGain(); // 메아리 효과를 얼마나 섞을지 정하는 조절기입니다.
  delayMix.gain.value = 0.18; // 메아리는 18% 크기로 섞입니다.

  convolver = audioCtx.createConvolver(); // 공간의 울림(잔향)을 입히는 장치입니다.
  convolver.buffer = createForestReverbIR(audioCtx); // 아까 만든 '숲속 울림 데이터'를 입힙니다.

  ambientGain = audioCtx.createGain(); // 배경음(바람 등) 전용 볼륨 조절기입니다.
  ambientGain.gain.value = 0.2; // 배경음은 20% 크기로 은은하게 들리게 합니다.

  dryGain.connect(masterGain); // 제 소리를 메인 조절기에 연결합니다.
  wetGain.connect(convolver); // 효과 적용 소리를 울림 장치에 연결합니다.
  convolver.connect(masterGain); // 울림 장치를 메인 조절기에 연결합니다.
  delayMix.connect(delayNode); // 메아리 통로를 메아리 장치에 연결합니다.
  delayNode.connect(delayFeedback); // 메아리 장치를 피드백에 연결합니다.
  delayFeedback.connect(delayNode); // 피드백을 다시 메아리 장치에 연결합니다. (반복 발생)
  delayNode.connect(masterGain); // 메아리 장치를 메인 조절기에 연결합니다.
  ambientGain.connect(masterGain); // 배경음 조절기를 메인 조절기에 연결합니다.
  masterGain.connect(limiter); // 메인 조절기를 리미터에 연결합니다.
  limiter.connect(audioCtx.destination); // 리미터를 최종 스피커에 연결합니다.

  return audioCtx; // 완성된 엔진을 돌려줍니다.
}

// 안전상의 이유로 잠겨있는 브라우저의 소리 기능을 사용자가 클릭했을 때 풀어주는 기능입니다.
export async function unlockAudioContext() {
  const ctx = ensureAudioContext(); // 소리 마법 엔진을 준비합니다.
  if (!ctx) return false; // 엔진이 없으면 실패입니다.

  if (ctx.state !== "running") { // 엔진이 잠겨(멈춰) 있다면
    try {
      await ctx.resume(); // 엔진을 다시 깨웁니다.
    } catch {
      return false; // 깨우기 실패하면 실패라고 알립니다.
    }
  }

  preloadSamples(); // 엔진이 깨어났으니 악기 소리들도 미리 불러옵니다.

  return ctx.state === "running"; // 성공적으로 켜졌는지 여부를 알려줍니다.
}

export function toggleSound() { // 소리를 켜고 끄는 스위치 기능입니다.
  const ctx = ensureAudioContext(); // 엔진 상태를 확인합니다.
  if (!ctx) return; // 엔진이 없으면 아무것도 안 합니다.

  if (ctx.state === "running") { // 소리가 나고 있다면
    stopAmbientLoop(); // 배경음을 먼저 멈춥니다.
    ctx.suspend(); // 엔진을 잠재웁니다.
    return;
  }

  ctx.resume(); // 소리가 꺼져 있었다면 엔진을 깨웁니다.
}

export function getAudioState() { // 현재 소리가 나고 있는지 상태를 물어보는 기능입니다.
  return { running: audioCtx?.state === "running" }; // 켜져 있으면 참(trun), 아니면 거짓(false)을 알려줍니다.
}

// 숲의 평화로운 배경음(바람소리, 새소리 등)을 계속 들려주는 기능입니다.
export function startAmbientLoop() {
  const ctx = ensureAudioContext(); // 엔진을 준비합니다.
  if (!ctx || ctx.state !== "running") return; // 엔진이 없거나 꺼져 있으면 중단합니다.

  clearScheduledTimers(); // 기존에 돌고 있던 시계들을 모두 멈춥니다.

  if (windNode) return; // 이미 바람 소리가 나고 있다면 또 켜지 않습니다.

  const src = ctx.createBufferSource(); // 소음 재생기를 만듭니다.
  src.buffer = createNoiseBuffer(ctx, 4.2, 0.6); // 4.2초짜리 긴 소음을 만듭니다.
  src.loop = true; // 이 소음이 끊기지 않고 계속 반복되게 합니다.

  const low = ctx.createBiquadFilter(); // 저음만 통과시키는 필터를 만듭니다.
  low.type = "lowpass"; // 높은 소리를 깎아서 웅웅거리는 느낌을 줍니다.
  low.frequency.value = 430; // 430Hz 이상의 소리를 걸러냅니다.

  const band = ctx.createBiquadFilter(); // 중간 대역만 남기는 필터를 만듭니다.
  band.type = "bandpass"; // 바람의 '휘잉~' 하는 느낌을 살립니다.
  band.frequency.value = 200; // 200Hz 주변 소리를 강조합니다.
  band.Q.value = 0.6; // 필터의 폭을 정합니다.

  const gain = ctx.createGain(); // 바람 볼륨 조절기를 만듭니다.
  gain.gain.value = 0.001; // 처음엔 거의 안 들리게 시작합니다.
  gain.gain.linearRampToValueAtTime(0.085, nowTime() + 1.2); // 1.2초 동안 서서히 소리를 키워 자연스럽게 등장시킵니다.

  src.connect(low); // 구성 요소들을 순서대로 연결합니다.
  low.connect(band);
  band.connect(gain);
  connectAmbient(gain); // 배경음 전용 통로에 연결합니다.

  src.start(); // 바람 소리 재생을 시작합니다.

  windNode = src; // 현재 바람 장치를 기록해둡니다.
  windGain = gain; // 현재 볼륨 조절기를 기록해둡니다.

  ambientTimer = window.setInterval(() => { // 1.9초마다 바람의 세기를 무작위로 바꿉니다.
    if (!audioCtx || audioCtx.state !== "running" || !windGain) return;
    const t = nowTime();
    const target = 0.045 + Math.random() * 0.07; // 새로운 목표 볼륨을 정합니다.
    windGain.gain.cancelScheduledValues(t); // 진행 중인 볼륨 변화를 취소합니다.
    windGain.gain.setValueAtTime(windGain.gain.value, t); // 현재 볼륨에서 시작하게 합니다.
    windGain.gain.linearRampToValueAtTime(target, t + 1.4 + Math.random()); // 1.4초 이상에 걸쳐 부드럽게 볼륨을 바꿉니다.
  }, 1900);

  scheduleBirdTone(); // 새소리도 주기적으로 나게 합니다.
  scheduleLeafDrop(); // 바스락 소리도 주기적으로 나게 합니다.
}

export function stopAmbientLoop() { // 배경음을 멈추는 기능입니다.
  clearScheduledTimers(); // 대기 중인 소리 시계들을 다 끕니다.

  if (windGain && audioCtx) { // 바람 소리가 나고 있다면
    const t = nowTime();
    windGain.gain.cancelScheduledValues(t);
    windGain.gain.setValueAtTime(windGain.gain.value, t);
    windGain.gain.linearRampToValueAtTime(0.001, t + 0.35); // 0.35초 동안 소리를 서서히 줄입니다.
    window.setTimeout(() => { // 소리가 다 줄어든 뒤에
      stopWindNode(); // 완전히 장치를 끕니다.
    }, 400);
    return;
  }

  stopWindNode(); // 바람 소리가 없었다면 그냥 장치만 확인 사살합니다.
}

// --- 아래는 각 식물(악기)을 터치했을 때 나는 개별 소리들입니다 ---

// '고슴도치 드럼' 소리를 냅니다. 쿵! 하는 소리가 나요.
export function playDrumMushroom() {
  if (playSample("drum", { wet: 0.16, delay: 0.06, gain: 1.06 })) return; // 미리 녹음된 드럼 소리가 있다면 그것을 연주합니다.

  const ctx = ensureAudioContext(); // 녹음된 소리가 없으면 직접 소리를 합성해서 만듭니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("drum", "synth");

  const t = nowTime();
  const kick = ctx.createOscillator(); // 쿵! 하는 저음을 위한 발생기입니다.
  const kickGain = ctx.createGain(); // 저음 볼륨 조절기입니다.

  kick.type = "sine"; // 가장 순수한 사인파를 사용합니다.
  kick.frequency.setValueAtTime(170, t); // 처음엔 약간 높은 음에서 시작해서
  kick.frequency.exponentialRampToValueAtTime(48, t + 0.11); // 0.11초 만에 아주 낮은 음으로 뚝 떨어뜨려 타격감을 줍니다.
  kickGain.gain.setValueAtTime(0.001, t);
  kickGain.gain.exponentialRampToValueAtTime(0.9, t + 0.009); // 아주 짧은 시간에 소리를 확 키웁니다.
  kickGain.gain.exponentialRampToValueAtTime(0.001, t + 0.22); // 0.22초 만에 소리를 사라지게 합니다.

  kick.connect(kickGain); // 발생기와 조절기를 연결합니다.
  connectSource(kickGain, 0.16, 0.05); // 효과 장치들에 연결합니다.
  kick.start(t); // 연주 시작!
  kick.stop(t + 0.24); // 0.24초 뒤에 완전 정지합니다.

  const click = ctx.createBufferSource(); // 드럼 스틱이 닿는 '탁' 소리를 위한 재생기입니다.
  const clickFilter = ctx.createBiquadFilter(); // '탁' 소리의 날카로움을 조절할 필터입니다.
  const clickGain = ctx.createGain(); // '탁' 소리 볼륨 조절기입니다.

  click.buffer = createNoiseBuffer(ctx, 0.04, 0.9); // 아주 짧은(0.04초) 강한 소음을 만듭니다.
  clickFilter.type = "bandpass"; // 중간 대역만 남겨 '탁' 소리처럼 들리게 합니다.
  clickFilter.frequency.value = 1300; // 1300Hz 주변을 강조합니다.
  clickFilter.Q.value = 0.8;
  clickGain.gain.setValueAtTime(0.24, t); // '탁' 소리는 적당히 작게 냅니다.
  clickGain.gain.exponentialRampToValueAtTime(0.001, t + 0.04); // 0.04초 만에 바로 사라지게 합니다.

  click.connect(clickFilter); // 구성 요소들을 연결합니다.
  clickFilter.connect(clickGain);
  connectSource(clickGain, 0.05, 0);
  click.start(t); // '탁' 소리 재생!
  click.stop(t + 0.05);
}

// '노래하는 백합(실로폰)' 소리를 냅니다. 도레미~ 하고 맑은 소리가 나요.
export function playXylophoneVine() {
  if (playSample("xylophone", { wet: 0.42, delay: 0.3, gain: 1.02 })) return; // 녹음된 실로폰 소리가 있으면 그것을 사용합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 맑은 소리를 합성합니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("xylophone", "synth");

  const sequence = [523.25, 659.25, 783.99]; // '도', '미', '솔' 음높이 목록입니다.
  const start = nowTime(); // 현재 시간을 잽니다.

  sequence.forEach((freq, idx) => { // 각 음을 순서대로 연주합니다.
    const t = start + idx * 0.06; // 음 사이의 간격을 0.06초로 둡니다. (아르페지오 느낌)

    const carrier = ctx.createOscillator(); // 기본 음을 내는 발생기입니다.
    const overtone = ctx.createOscillator(); // 맑은 느낌을 더해줄 높은 배음 발생기입니다.
    const gain = ctx.createGain(); // 볼륨 조절기입니다.

    carrier.type = "triangle"; // 부드러운 삼각형 파동을 사용합니다.
    overtone.type = "sine"; // 맑은 사인 파동을 사용합니다.
    carrier.frequency.setValueAtTime(freq, t); // 기본 주파수를 설정합니다.
    overtone.frequency.setValueAtTime(freq * 2.02, t); // 두 배보다 살짝 높은 주파수로 신비감을 줍니다.

    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.3, t + 0.012); // 아주 빠르게 소리를 키워 '팅~' 하는 느낌을 줍니다.
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.26); // 0.26초 동안 소리가 여운을 남기며 사라집니다.

    carrier.connect(gain); // 발생기들을 조절기에 연결합니다.
    overtone.connect(gain);
    connectSource(gain, 0.44, 0.3); // 울림과 메아리 효과를 듬뿍 줍니다.

    carrier.start(t); // 연주 시작!
    overtone.start(t);
    carrier.stop(t + 0.28); // 0.28초 뒤에 멈춥니다.
    overtone.stop(t + 0.25);
  });
}

// '행운 클로버(탬버린)' 소리를 냅니다. 챵~ 하는 경쾌한 소리가 나요.
export function playTambourineFlower() {
  if (playSample("tambourine", { wet: 0.2, delay: 0.08, gain: 1.08 })) return; // 녹음된 탬버린 소리가 있으면 그것을 연주합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 짤랑거리는 소리를 만듭니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("tambourine", "synth");

  const t = nowTime();
  const src = ctx.createBufferSource(); // 소음 재생기를 만듭니다.
  const high = ctx.createBiquadFilter(); // 아주 높은 소리만 남길 필터입니다.
  const ring = ctx.createBiquadFilter(); // 특정 금속성 소리를 강조할 필터입니다.
  const gain = ctx.createGain(); // 볼륨 조절기입니다.

  src.buffer = createNoiseBuffer(ctx, 0.12, 0.75); // 0.12초짜리 짧은 소음을 사용합니다.
  high.type = "highpass"; // 저음을 다 깎아버립니다.
  high.frequency.value = 2600; // 2600Hz 이상의 고음만 남깁니다.
  ring.type = "bandpass"; // 쇳소리가 나는 부분을 강조합니다.
  ring.frequency.value = 5200; // 5200Hz 주변을 강조해서 짤랑거리는 느낌을 줍니다.
  ring.Q.value = 1.4; // 강조할 폭을 좁게 설정합니다.

  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.42, t + 0.012); // 아주 빠르게 소리를 키워 타격감을 줍니다.
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14); // 0.14초 만에 소리를 사라지게 합니다.

  src.connect(high); // 필터와 조절기를 순서대로 연결합니다.
  high.connect(ring);
  ring.connect(gain);
  connectSource(gain, 0.24, 0.1); // 약간의 울림을 섞어줍니다.

  src.start(t); // 재생 시작!
  src.stop(t + 0.16);
}

export function playFistBeat() { // 주먹을 쥐었을 때 나는 묵직한 타격음입니다.
  if (playSample("fist", { wet: 0.12, delay: 0.05, gain: 1.1 })) return; // 녹음된 주먹 소리가 있으면 사용합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 쿵쿵거리는 소리를 만듭니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("fist", "synth");

  const t = nowTime();
  const osc = ctx.createOscillator(); // 묵직한 파동 발생기입니다.
  const gain = ctx.createGain(); // 볼륨 조절기입니다.

  osc.type = "square"; // 각진 사각형 파동을 사용해 거친 소리를 냅니다.
  osc.frequency.setValueAtTime(126, t); // 엔진 소리 같은 낮은 음에서 시작해
  osc.frequency.exponentialRampToValueAtTime(58, t + 0.1); // 더 낮은 음으로 뚝 떨어뜨립니다.
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.6, t + 0.012); // 순간적으로 소리를 키워 '퍽' 하는 느낌을 줍니다.
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2); // 0.2초 만에 소리를 끕니다.

  osc.connect(gain); // 발생기와 조절기를 연결합니다.
  connectSource(gain, 0.12, 0.05); // 조절기를 효과 장치들에 연결합니다.
  osc.start(t); // 재생!
  osc.stop(t + 0.22);
}

// '마법 고사리' 소리를 냅니다. 링~ 하는 신비로운 종소리가 나요.
export function playPinkyChime() {
  if (playSample("pinky", { wet: 0.45, delay: 0.3, gain: 1.02 })) return; // 녹음된 고사리 소리가 있으면 사용합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 종소리를 합성합니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("pinky", "synth");

  const start = nowTime();
  [987.77, 1174.66].forEach((freq, idx) => { // 두 가지 높은 음을 순서대로 냅니다.
    const t = start + idx * 0.04;
    [ // 각 음에 여러 배음을 섞어 은은한 금속 소리를 만듭니다.
      { ratio: 1, vol: 0.24 }, // 기본음
      { ratio: 2.5, vol: 0.11 }, // 높은 배음 1
      { ratio: 5.1, vol: 0.05 }  // 높은 배음 2
    ].forEach((harmonic) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine"; // 맑은 사인파를 사용합니다.
      osc.frequency.setValueAtTime(freq * harmonic.ratio, t); // 배음 비율에 맞춰 주파수를 정합니다.
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.linearRampToValueAtTime(harmonic.vol, t + 0.02); // 부드럽게 소리를 키웁니다.
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.32); // 0.32초 동안 사라지게 합니다.

      osc.connect(gain);
      connectSource(gain, 0.45, 0.3); // 울림 효과를 많이 줍니다.

      osc.start(t); // 재생!
      osc.stop(t + 0.34);
    });
  });
}

// '아기 사슴' 소리를 냅니다. 삐요~ 하고 귀여운 소리가 나요.
// '아기 사슴' 소리를 냅니다. 삐요~ 하고 귀여운 소리가 나요.
export function playHeartBloom() {
  if (playSample("heart", { wet: 0.52, delay: 0.34, gain: 1.02 })) return; // 녹음된 사슴 소리가 있으면 사용합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 귀여운 소리를 합성합니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("heart", "synth");

  const t = nowTime();
  [392, 395].forEach((freq) => { // 두 개의 비슷한 음을 섞어서 풍성하게 만듭니다.
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const lfo = ctx.createOscillator(); // 소리를 '우아우아' 하게 떨리게 할 진동기입니다.
    const lfoGain = ctx.createGain(); // 떨림의 강도를 조절합니다.

    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, t);
    osc.frequency.linearRampToValueAtTime(freq * 1.335, t + 0.2); // 0.2초 동안 음높이를 살짝 올립니다. (삐요~)

    lfo.type = "sine"; // 부드러운 진동을 사용합니다.
    lfo.frequency.value = 5.5; // 1초에 5.5번 떨리게 합니다.
    lfoGain.gain.value = 4; // 떨림의 폭을 설정합니다.
    lfo.connect(lfoGain);
    lfoGain.connect(osc.frequency); // 진동기를 음높이에 연결해 비브라토 효과를 줍니다.

    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.22, t + 0.05); // 0.05초 만에 소리가 나게 합니다.
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5); // 0.5초 동안 서서히 사라집니다.

    osc.connect(gain);
    connectSource(gain, 0.52, 0.36); // 울림과 메아리를 많이 줍니다.

    lfo.start(t);
    osc.start(t);
    lfo.stop(t + 0.5);
    osc.stop(t + 0.52);
  });
}

export function playAnimalRoll() { // 특별한 '동물 구르기' 소리를 냅니다.
  if (Math.random() < 0.5) {
    if (playSample("animalCat", { wet: 0.24, delay: 0.18, gain: 1.06 })) return;
    if (playSample("animalDog", { wet: 0.24, delay: 0.18, gain: 1.06 })) return;
  } else {
    if (playSample("animalDog", { wet: 0.24, delay: 0.18, gain: 1.06 })) return;
    if (playSample("animalCat", { wet: 0.24, delay: 0.18, gain: 1.06 })) return;
  }

  if (playSample("animal", { wet: 0.28, delay: 0.22, gain: 1.05 })) return; // 녹음된 소리가 있으면 사용합니다.

  const ctx = ensureAudioContext(); // 소리가 없으면 직접 합성합니다.
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("animal", "synth");

  const start = nowTime();
  [196, 246.94, 293.66, 392].forEach((freq, idx) => { // 여러 음을 계단식으로 빠르게 연주합니다.
    const t = start + idx * 0.05; // 0.05초 간격으로 소리가 납니다.
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sawtooth"; // 톱니 모양 파동을 사용해 징징거리는 독특한 소리를 만듭니다.
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.2, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14); // 짧게 끊어서 연주합니다.

    osc.connect(gain);
    connectSource(gain, 0.28, 0.22);

    osc.start(t); // 재생!
    osc.stop(t + 0.16);
  });
}

// --- 아래는 아이들을 위한 6가지 새로운 멜로디 악기입니다 ---
// 모든 악기는 C 메이저 스케일 기반으로 조화롭게 연주됩니다.

// 모든 악기가 "같은 음악 흐름" 안에서 움직이도록 공통 코드 진행을 정의합니다.
// 시간이 지나면 C -> Am -> F -> G 순서로 넘어가며, 각 악기는 현재 코드에 어울리는 음만 고르게 됩니다.
// 코드가 너무 빨리 바뀌면 멜로디가 안정되기 전에 넘어가므로, 조금 더 천천히 진행시킵니다.
const HARMONY_SECTION_SEC = 3.2;
const HARMONY_PROGRESSION = [
  {
    name: "C",
    drum: [130.81, 130.81, 196.0, 164.81],
    tambourine: [659.25, 659.25, 783.99, 659.25],
    xylophone: [523.25, 659.25, 587.33, 783.99, 659.25],
    whistle: [523.25, 587.33, 659.25, 587.33],
    triangle: [783.99, 1046.5, 783.99]
  },
  {
    name: "Am",
    drum: [110.0, 110.0, 164.81, 220.0],
    tambourine: [659.25, 659.25, 880.0, 659.25],
    xylophone: [659.25, 880.0, 783.99, 987.77, 880.0],
    whistle: [659.25, 783.99, 880.0, 783.99],
    triangle: [880.0, 1046.5, 880.0]
  },
  {
    name: "F",
    drum: [87.31, 87.31, 130.81, 174.61],
    tambourine: [698.46, 698.46, 783.99, 698.46],
    xylophone: [698.46, 783.99, 659.25, 880.0, 783.99],
    whistle: [698.46, 783.99, 880.0, 783.99],
    triangle: [698.46, 880.0, 698.46]
  },
  {
    name: "G",
    drum: [98.0, 98.0, 146.83, 196.0],
    tambourine: [783.99, 783.99, 987.77, 783.99],
    xylophone: [783.99, 987.77, 880.0, 783.99, 698.46],
    whistle: [783.99, 880.0, 987.77, 880.0],
    triangle: [783.99, 987.77, 783.99]
  }
];

// 각 악기가 연주할 때마다 다음 음으로 진행하기 위한 카운터
let drumBeatIndex = 0;
let tambourineBeatIndex = 0;
let xyloBeatIndex = 0;
let whistleBeatIndex = 0;
let triangleBeatIndex = 0;

function getCurrentHarmonyFrame() {
  const ctx = ensureAudioContext();
  if (!ctx) return HARMONY_PROGRESSION[0];
  const sectionIndex = Math.floor(ctx.currentTime / HARMONY_SECTION_SEC) % HARMONY_PROGRESSION.length;
  return HARMONY_PROGRESSION[sectionIndex];
}

function getAutoHarmonyFrequency(trackName, beatIndex) {
  const frame = getCurrentHarmonyFrame();
  const pattern = frame?.[trackName];
  if (!Array.isArray(pattern) || pattern.length === 0) return null;
  return pattern[beatIndex % pattern.length];
}

// 1. 쿵 (북/드럼) - 심장 박동처럼 일정한 중심 박자 (안정감)
// 베이스 라인: C - C - G - G - A - A - G (도도솔솔라라솔)
export function playKids_Drum() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_drum", "synth");

  // 현재 코드 진행에 맞는 베이스 음을 선택합니다.
  const freq = getAutoHarmonyFrequency("drum", drumBeatIndex) || 130.81;
  drumBeatIndex++;

  const t = reserveInstrumentStart("drum");
  const kick = ctx.createOscillator();
  const kickGain = ctx.createGain();

  kick.type = "sine";
  kick.frequency.setValueAtTime(freq * 1.3, t);
  kick.frequency.exponentialRampToValueAtTime(freq, t + 0.08);

  kickGain.gain.setValueAtTime(0.001, t);
  kickGain.gain.exponentialRampToValueAtTime(0.72, t + 0.01);
  kickGain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);

  kick.connect(kickGain);
  connectSource(kickGain, 0.08, 0.04);
  kick.start(t);
  kick.stop(t + 0.42);
}

// 2. 챵 (탬버린/심벌즈) - 화려한 리듬, 에너지
// note 값을 직접 주면 그 음을, 생략하면 현재 코드 진행에 맞는 음을 자동으로 고릅니다.
export function playKids_Tambourine(note) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_tambourine", "synth");

  const scale = [523.25, 587.33, 659.25, 698.46, 783.99]; // C5, D5, E5, F5, G5
  const hasExplicitNote = Number.isFinite(note);
  const freq = hasExplicitNote
    ? scale[note % scale.length]
    : (getAutoHarmonyFrequency("tambourine", tambourineBeatIndex) || scale[0]);
  if (!hasExplicitNote) tambourineBeatIndex++;
  const t = reserveInstrumentStart("tambourine");

  // 메탈릭한 소리를 위한 노이즈
  const src = ctx.createBufferSource();
  const high = ctx.createBiquadFilter();
  const ring = ctx.createBiquadFilter();
  const gain = ctx.createGain();

  src.buffer = createNoiseBuffer(ctx, 0.15, 0.8);
  high.type = "highpass";
  high.frequency.value = 3000;
  ring.type = "bandpass";
  ring.frequency.value = freq * 1.75;
  ring.Q.value = 2.5;

  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.3, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);

  src.connect(high);
  high.connect(ring);
  ring.connect(gain);
  connectSource(gain, 0.12, 0.06);

  src.start(t);
  src.stop(t + 0.2);
}

// 3. 도레미 (실로폰/건반) - 예쁜 멜로디, 선율
// 메인 게임에서 호출할 때는 note 없이 호출되어 자동 멜로디가 흐르도록 설계했습니다.
export function playKids_Xylophone(note) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_xylophone", "synth");

  const scale = [523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77]; // C5-B5
  const hasExplicitNote = Number.isFinite(note);
  const freq = hasExplicitNote
    ? scale[note % scale.length]
    : (getAutoHarmonyFrequency("xylophone", xyloBeatIndex) || scale[0]);
  if (!hasExplicitNote) xyloBeatIndex++;
  const t = reserveInstrumentStart("xylophone");

  const carrier = ctx.createOscillator();
  const modulator = ctx.createOscillator();
  const gain = ctx.createGain();

  carrier.type = "sine";
  modulator.type = "sine";
  carrier.frequency.setValueAtTime(freq, t);
  modulator.frequency.setValueAtTime(freq * 2.4, t); // 배음을 조금 줄여 덜 날카롭게 만듭니다.

  gain.gain.setValueAtTime(0.001, t);
  gain.gain.linearRampToValueAtTime(0.28, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.36);

  carrier.connect(gain);
  modulator.connect(gain);
  connectSource(gain, 0.2, 0.12);

  carrier.start(t);
  modulator.start(t);
  carrier.stop(t + 0.38);
  modulator.stop(t + 0.38);
}

// 4. 휘파람 (리코더/플루트) - 높고 맑은 포인트 멜로디
// 메인 게임에서는 카운터 멜로디 역할로 자동 음 선택을 합니다.
export function playKids_Whistle(note) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_whistle", "synth");

  const scale = [523.25, 587.33, 659.25, 698.46, 783.99]; // C5, D5, E5, F5, G5
  const hasExplicitNote = Number.isFinite(note);
  const freq = hasExplicitNote
    ? scale[note % scale.length]
    : (getAutoHarmonyFrequency("whistle", whistleBeatIndex) || scale[0]);
  if (!hasExplicitNote) whistleBeatIndex++;
  const t = reserveInstrumentStart("whistle");

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const lfo = ctx.createOscillator(); // 비브라토
  const lfoGain = ctx.createGain();

  osc.type = "sine";
  osc.frequency.setValueAtTime(freq, t);

  // 휘파람 특유의 떨림(비브라토)
  lfo.type = "sine";
  lfo.frequency.value = 4.2;
  lfoGain.gain.value = 4.5;
  lfo.connect(lfoGain);
  lfoGain.connect(osc.frequency);

  gain.gain.setValueAtTime(0.001, t);
  gain.gain.linearRampToValueAtTime(0.18, t + 0.06);
  gain.gain.linearRampToValueAtTime(0.15, t + 0.24);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.42);

  osc.connect(gain);
  connectSource(gain, 0.24, 0.14);

  lfo.start(t);
  osc.start(t);
  lfo.stop(t + 0.44);
  osc.stop(t + 0.44);
}

// 5. 🐾 랜덤 동물 소리 (Wildcard) - 깜짝 재미 요소
// 매번 다른 동물 소리 (무작위 음정, 재미있는 효과)
export function playKids_AnimalSurprise() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_animal", "synth");

  const animalTypes = [
    { name: "고양이", freqs: [400, 800, 600] },
    { name: "강아지", freqs: [220, 180, 200] },
    { name: "새", freqs: [1200, 1400, 1300] },
    { name: "개구리", freqs: [150, 180, 150] }
  ];

  const animal = animalTypes[Math.floor(Math.random() * animalTypes.length)];
  const t = reserveInstrumentStart("animal");

  animal.freqs.forEach((freq, idx) => {
    const delay = t + idx * 0.12;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = idx === 0 ? "square" : "triangle";
    osc.frequency.setValueAtTime(freq, delay);
    osc.frequency.linearRampToValueAtTime(freq * 0.9, delay + 0.1);

    gain.gain.setValueAtTime(0.001, delay);
    gain.gain.linearRampToValueAtTime(0.26, delay + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, delay + 0.15);

    osc.connect(gain);
    connectSource(gain, 0.18, 0.08);

    osc.start(delay);
    osc.stop(delay + 0.17);
  });
}

// 6. 반짝 (트라이앵글/종) - 신비로움, 전환
// 위쪽 하모니를 채워주는 보조 멜로디 역할입니다.
export function playKids_Triangle(note) {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;
  emitSoundPlayed("kids_triangle", "synth");

  const scale = [698.46, 783.99, 880.0, 987.77, 1046.5, 1174.66]; // F5-D6
  const hasExplicitNote = Number.isFinite(note);
  const baseFreq = hasExplicitNote
    ? scale[note % scale.length]
    : (getAutoHarmonyFrequency("triangle", triangleBeatIndex) || scale[0]);
  if (!hasExplicitNote) triangleBeatIndex++;
  const t = reserveInstrumentStart("triangle");

  // 종소리는 여러 배음이 함께 울립니다
  [
    { ratio: 1, vol: 0.18 },
    { ratio: 2.4, vol: 0.08 },
    { ratio: 4.8, vol: 0.04 }
  ].forEach((harmonic) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(baseFreq * harmonic.ratio, t);

    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(harmonic.vol, t + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.52);

    osc.connect(gain);
    connectSource(gain, 0.26, 0.12);

    osc.start(t);
    osc.stop(t + 0.54);
  });
}

// ========================================
// 멜로디 시퀀스 시스템 (손동작 유지 시 연속 재생)
// ========================================

let melodyTimers = {}; // 각 제스처별 타이머 저장
let melodyIndices = {}; // 각 제스처별 현재 음표 인덱스

// 멜로디 패턴 정의 (각 악기별로 다른 멜로디)
const melodyPatterns = {
  drum: [0, 1, 2, 1, 0, 2, 3, 2], // 쿵 드럼 패턴
  xylophone: [0, 2, 4, 2, 0, 4, 6, 4], // 도레미 실로폰 패턴 (도-미-솔-미...)
  tambourine: [1, 3, 1, 4, 1, 3, 2, 0], // 챵 탬버린 패턴
  whistle: [4, 3, 2, 3, 4, 4, 4, 3], // 휘파람 패턴
  triangle: [0, 2, 4, 5, 4, 2, 0, 1], // 반짝 트라이앵글 패턴
};

// 멜로디 시퀀스 시작 함수
export function startMelodySequence(instrumentType, playFunction) {
  // 이미 재생 중이면 무시
  if (melodyTimers[instrumentType]) return;

  // 인덱스 초기화
  melodyIndices[instrumentType] = 0;

  const pattern = melodyPatterns[instrumentType] || [0, 1, 2, 3, 4];
  const interval = 300; // 300ms마다 다음 음표 재생

  // 첫 음표 즉시 재생
  const note = pattern[melodyIndices[instrumentType]];
  playFunction(note);
  melodyIndices[instrumentType] = (melodyIndices[instrumentType] + 1) % pattern.length;

  // 타이머 시작
  melodyTimers[instrumentType] = setInterval(() => {
    const note = pattern[melodyIndices[instrumentType]];
    playFunction(note);
    melodyIndices[instrumentType] = (melodyIndices[instrumentType] + 1) % pattern.length;
  }, interval);
}

// 멜로디 시퀀스 중지 함수
export function stopMelodySequence(instrumentType) {
  if (melodyTimers[instrumentType]) {
    clearInterval(melodyTimers[instrumentType]);
    melodyTimers[instrumentType] = null;
    melodyIndices[instrumentType] = 0;
  }
}

// 모든 멜로디 중지
export function stopAllMelodies() {
  Object.keys(melodyTimers).forEach((type) => {
    stopMelodySequence(type);
  });
}

// 현재 재생 중인지 확인
export function isMelodyPlaying(instrumentType) {
  return !!melodyTimers[instrumentType];
}
