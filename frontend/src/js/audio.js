let audioCtx = null;
let masterGain = null;
let ambientTimer = null;
let limiter = null;

function nowTime() {
  return audioCtx ? audioCtx.currentTime : 0;
}

export function ensureAudioContext() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audioCtx = new Ctx();

    limiter = audioCtx.createDynamicsCompressor();
    limiter.threshold.value = -16;
    limiter.knee.value = 18;
    limiter.ratio.value = 4;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.2;

    masterGain = audioCtx.createGain();
    masterGain.gain.value = 1.42;
    masterGain.connect(limiter);
    limiter.connect(audioCtx.destination);
  }

  return audioCtx;
}

function connectSource(node) {
  if (!masterGain) return;
  node.connect(masterGain);
}

export async function unlockAudioContext() {
  const ctx = ensureAudioContext();
  if (!ctx) return false;

  if (ctx.state !== "running") {
    try {
      await ctx.resume();
    } catch {
      return false;
    }
  }

  return ctx.state === "running";
}

export function toggleSound() {
  const ctx = ensureAudioContext();
  if (!ctx) return;

  if (ctx.state === "running") ctx.suspend();
  else ctx.resume();
}

export function getAudioState() {
  return {
    running: audioCtx?.state === "running"
  };
}

export function playDrumMushroom() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const t = nowTime();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = "triangle";
  osc.frequency.setValueAtTime(172, t);
  osc.frequency.exponentialRampToValueAtTime(62, t + 0.09);
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.76, t + 0.014);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.17);

  osc.connect(gain);
  connectSource(gain);
  osc.start(t);
  osc.stop(t + 0.2);

  const noiseBuffer = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.06), ctx.sampleRate);
  const channel = noiseBuffer.getChannelData(0);
  for (let i = 0; i < channel.length; i += 1) {
    channel[i] = (Math.random() * 2 - 1) * (1 - i / channel.length);
  }

  const noise = ctx.createBufferSource();
  const filter = ctx.createBiquadFilter();
  const noiseGain = ctx.createGain();
  noise.buffer = noiseBuffer;
  filter.type = "bandpass";
  filter.frequency.value = 900;
  filter.Q.value = 0.8;
  noiseGain.gain.setValueAtTime(0.28, t);
  noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
  noise.connect(filter).connect(noiseGain);
  connectSource(noiseGain);
  noise.start(t);
  noise.stop(t + 0.08);
}

export function playXylophoneVine() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const sequence = [523.25, 659.25, 783.99];
  const start = nowTime();

  sequence.forEach((freq, idx) => {
    const t = start + idx * 0.06;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.27, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    osc.connect(gain);
    connectSource(gain);
    osc.start(t);
    osc.stop(t + 0.22);
  });
}

export function playTambourineFlower() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const t = nowTime();
  const noiseBuffer = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.12), ctx.sampleRate);
  const data = noiseBuffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = (Math.random() * 2 - 1) * 0.7;
  }

  const src = ctx.createBufferSource();
  const high = ctx.createBiquadFilter();
  const gain = ctx.createGain();

  src.buffer = noiseBuffer;
  high.type = "highpass";
  high.frequency.value = 2800;

  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.38, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);

  src.connect(high).connect(gain);
  connectSource(gain);
  src.start(t);
  src.stop(t + 0.16);
}

export function playFistBeat() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const t = nowTime();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = "square";
  osc.frequency.setValueAtTime(130, t);
  osc.frequency.exponentialRampToValueAtTime(58, t + 0.1);
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.exponentialRampToValueAtTime(0.58, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);

  osc.connect(gain);
  connectSource(gain);
  osc.start(t);
  osc.stop(t + 0.2);
}

export function playPinkyChime() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const start = nowTime();
  [987.77, 1174.66].forEach((freq, idx) => {
    const t = start + idx * 0.04;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.24, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.26);
    osc.connect(gain);
    connectSource(gain);
    osc.start(t);
    osc.stop(t + 0.3);
  });
}

export function playHeartBloom() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const t = nowTime();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(392, t);
  osc.frequency.linearRampToValueAtTime(523.25, t + 0.18);
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.linearRampToValueAtTime(0.3, t + 0.04);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.42);
  osc.connect(gain);
  connectSource(gain);
  osc.start(t);
  osc.stop(t + 0.45);
}

export function playAnimalRoll() {
  const ctx = ensureAudioContext();
  if (!ctx || ctx.state !== "running") return;

  const start = nowTime();
  [196, 246.94, 293.66, 392].forEach((freq, idx) => {
    const t = start + idx * 0.05;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.001, t);
    gain.gain.linearRampToValueAtTime(0.18, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
    osc.connect(gain);
    connectSource(gain);
    osc.start(t);
    osc.stop(t + 0.14);
  });
}

function playAmbientNote(freq, len = 0.5, volume = 0.12) {
  if (!audioCtx || audioCtx.state !== "running") return;

  const t = nowTime();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "triangle";
  osc.frequency.setValueAtTime(freq, t);
  gain.gain.setValueAtTime(0.001, t);
  gain.gain.linearRampToValueAtTime(volume, t + 0.08);
  gain.gain.exponentialRampToValueAtTime(0.001, t + len);
  osc.connect(gain);
  connectSource(gain);
  osc.start(t);
  osc.stop(t + len + 0.04);
}

export function startAmbientLoop() {
  if (ambientTimer) return;

  const phrase = [261.63, 293.66, 329.63, 349.23, 293.66];
  let idx = 0;

  ambientTimer = setInterval(() => {
    if (!audioCtx || audioCtx.state !== "running") return;
    playAmbientNote(phrase[idx % phrase.length]);
    idx += 1;
  }, 920);
}

export function stopAmbientLoop() {
  if (!ambientTimer) return;
  clearInterval(ambientTimer);
  ambientTimer = null;
}
