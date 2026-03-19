let audioCtx = null;
let masterGain = null;
let masterLimiter = null;
let ambientGain = null;
let reverbInputGain = null;
let reverbWetGain = null;
let reverbNode = null;
let delayInputGain = null;
let delayNode = null;
let delayFeedbackGain = null;
let delayToneFilter = null;
let delayWetGain = null;
let soundEnabled = true;
let unlocked = false;
let playbackContext = null;
let activeVoiceCount = 0;

const activeMelodies = new Map();
const melodyTickMs = 300;
const MASTER_TARGET_GAIN = 0.82;
const REVERB_SEND_DEFAULT = 0.12;
const DELAY_SEND_DEFAULT = 0.04;
const MAX_ACTIVE_VOICES = 24;
const COMPANION_VOICE_THRESHOLD = 14;
const ADAPTIVE_MELODY_SLOWDOWN_THRESHOLD = 16;

const NOTE_INDEX = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11
};

const melodyPalettes = {
  drum: [130.81, 123.47, 110.0, 98.0, 87.31],
  piano: [261.63, 293.66, 329.63, 392.0, 440.0, 523.25],
  guitar: [196.0, 220.0, 246.94, 293.66, 329.63, 392.0],
  flute: [523.25, 587.33, 659.25, 698.46, 783.99, 880.0],
  violin: [392.0, 440.0, 493.88, 523.25, 587.33, 659.25],
  bell: [783.99, 987.77, 1174.66, 1318.51, 1567.98],

  // 하위 호환 키
  xylophone: [261.63, 293.66, 329.63, 392.0, 440.0, 523.25],
  tambourine: [196.0, 220.0, 246.94, 293.66, 329.63, 392.0],
  whistle: [523.25, 587.33, 659.25, 698.46, 783.99, 880.0],
  triangle: [783.99, 987.77, 1174.66, 1318.51, 1567.98],
  animal: [392.0, 440.0, 523.25, 587.33]
};

const melodyPhrases = {
  drum: [
    [130.81, 110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 110.0, 98.0, 87.31, 98.0, 110.0, 130.81, 123.47, 110.0, 98.0],
    [110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 130.81, 110.0, 98.0, 87.31],
    [130.81, 123.47, 110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 130.81, 110.0],
    [98.0, 87.31, 98.0, 110.0, 123.47, 110.0, 98.0, 87.31, 98.0, 110.0, 123.47, 130.81, 123.47, 110.0, 98.0, 87.31]
  ],
  piano: [
    [261.63, 329.63, 392.0, 440.0, 392.0, 329.63, 293.66, 261.63, 293.66, 329.63, 392.0, 523.25, 440.0, 392.0, 329.63, 293.66],
    [293.66, 349.23, 440.0, 523.25, 440.0, 349.23, 329.63, 293.66, 329.63, 349.23, 440.0, 587.33, 523.25, 440.0, 349.23, 329.63],
    [261.63, 293.66, 329.63, 392.0, 440.0, 392.0, 329.63, 293.66, 261.63, 293.66, 329.63, 392.0, 523.25, 440.0, 392.0, 329.63],
    [329.63, 392.0, 440.0, 523.25, 440.0, 392.0, 349.23, 329.63, 349.23, 392.0, 440.0, 659.25, 523.25, 440.0, 392.0, 349.23],
    [261.63, 329.63, 440.0, 392.0, 329.63, 293.66, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 392.0, 329.63, 293.66, 261.63]
  ],
  guitar: [
    [196.0, 220.0, 246.94, 293.66, 246.94, 220.0, 196.0, 220.0, 246.94, 293.66, 329.63, 293.66, 246.94, 220.0, 196.0, 220.0],
    [220.0, 246.94, 293.66, 329.63, 293.66, 246.94, 220.0, 196.0, 220.0, 246.94, 293.66, 392.0, 329.63, 293.66, 246.94, 220.0],
    [196.0, 246.94, 293.66, 329.63, 293.66, 246.94, 220.0, 196.0, 220.0, 246.94, 293.66, 329.63, 392.0, 329.63, 293.66, 246.94],
    [220.0, 196.0, 220.0, 246.94, 293.66, 329.63, 293.66, 246.94, 220.0, 246.94, 293.66, 329.63, 440.0, 392.0, 329.63, 293.66]
  ],
  flute: [
    [523.25, 587.33, 659.25, 698.46, 783.99, 698.46, 659.25, 587.33, 523.25, 587.33, 659.25, 783.99, 880.0, 783.99, 698.46, 659.25],
    [587.33, 659.25, 698.46, 783.99, 880.0, 783.99, 698.46, 659.25, 587.33, 659.25, 783.99, 880.0, 987.77, 880.0, 783.99, 698.46],
    [523.25, 659.25, 698.46, 783.99, 698.46, 659.25, 587.33, 523.25, 587.33, 659.25, 698.46, 783.99, 880.0, 783.99, 698.46, 659.25],
    [659.25, 698.46, 783.99, 880.0, 987.77, 880.0, 783.99, 698.46, 659.25, 783.99, 880.0, 987.77, 1046.5, 987.77, 880.0, 783.99]
  ],
  violin: [
    [392.0, 440.0, 493.88, 523.25, 587.33, 523.25, 493.88, 440.0, 392.0, 440.0, 493.88, 587.33, 659.25, 587.33, 523.25, 493.88],
    [440.0, 493.88, 523.25, 587.33, 659.25, 587.33, 523.25, 493.88, 440.0, 493.88, 523.25, 659.25, 698.46, 659.25, 587.33, 523.25],
    [392.0, 493.88, 523.25, 587.33, 523.25, 493.88, 440.0, 392.0, 440.0, 493.88, 523.25, 587.33, 659.25, 587.33, 523.25, 493.88],
    [493.88, 523.25, 587.33, 659.25, 698.46, 659.25, 587.33, 523.25, 493.88, 523.25, 587.33, 659.25, 783.99, 698.46, 659.25, 587.33]
  ],
  bell: [
    [783.99, 987.77, 1174.66, 987.77, 1318.51, 1174.66, 987.77, 783.99, 987.77, 1174.66, 1318.51, 1567.98, 1318.51, 1174.66, 987.77, 783.99],
    [987.77, 1174.66, 1318.51, 1174.66, 1567.98, 1318.51, 1174.66, 987.77, 1174.66, 1318.51, 1567.98, 1760.0, 1567.98, 1318.51, 1174.66, 987.77],
    [783.99, 1174.66, 1318.51, 1567.98, 1318.51, 1174.66, 987.77, 783.99, 987.77, 1174.66, 1318.51, 1567.98, 1760.0, 1567.98, 1318.51, 1174.66],
    [987.77, 1318.51, 1567.98, 1318.51, 1760.0, 1567.98, 1318.51, 1174.66, 987.77, 1174.66, 1318.51, 1567.98, 1975.53, 1760.0, 1567.98, 1318.51]
  ]
};

const melodyTranspositions = {
  drum: [0],
  piano: [0, 2, -2, 5],
  guitar: [0, 2, -3, 4],
  flute: [0, 2, 4, -2],
  violin: [0, 2, -2, 5],
  bell: [0, 2, 5]
};

const MELODY_ALIAS = {
  xylophone: "piano",
  tambourine: "guitar",
  whistle: "flute",
  triangle: "bell"
};

const melodyCursor = new Map();

function normalizeMelodyName(name) {
  return MELODY_ALIAS[name] || name;
}

function getMelodyStateKey(baseKey, handKey = null) {
  const resolvedHand = handKey == null
    ? String(playbackContext?.handKey || "default").toLowerCase()
    : String(handKey || "default").toLowerCase();
  return `${baseKey}:${resolvedHand}`;
}

function getMelodyState(name, handKey = null) {
  const baseKey = normalizeMelodyName(name);
  const stateKey = getMelodyStateKey(baseKey, handKey);
  if (!melodyCursor.has(stateKey)) {
    const phraseCount = Math.max(1, (melodyPhrases[baseKey] || []).length);
    melodyCursor.set(stateKey, {
      phraseIndex: Math.floor(Math.random() * phraseCount),
      transposeSemitone: 0,
      lastStep: -1,
      freeStep: 0
    });
  }
  return melodyCursor.get(stateKey);
}

function pickDifferentOption(options, currentValue) {
  if (!Array.isArray(options) || options.length === 0) return currentValue;
  if (options.length === 1) return options[0];

  let next = options[Math.floor(Math.random() * options.length)];
  if (next === currentValue) {
    const idx = options.indexOf(next);
    next = options[(idx + 1) % options.length];
  }
  return next;
}

function transposeFrequency(value, semitone) {
  if (!Number.isFinite(value)) return value;
  if (!Number.isFinite(semitone) || semitone === 0) return value;
  const shifted = value * Math.pow(2, semitone / 12);
  return Math.min(2600, Math.max(70, shifted));
}

function advanceMelodyVariation(baseKey, state) {
  const phrases = melodyPhrases[baseKey] || [];
  if (phrases.length > 1) {
    const phraseChoices = phrases.map((_, index) => index);
    state.phraseIndex = pickDifferentOption(phraseChoices, state.phraseIndex);
  }

  const transposeChoices = melodyTranspositions[baseKey] || [0];
  state.transposeSemitone = pickDifferentOption(transposeChoices, state.transposeSemitone);
}

function chooseFromPalette(name, note) {
  const baseKey = normalizeMelodyName(name);
  const palette = melodyPalettes[baseKey] || melodyPalettes.piano;
  const phrases = melodyPhrases[baseKey] || null;

  if (Array.isArray(note) && note.length) return note;
  if (typeof note === "string") return note;

  const state = getMelodyState(baseKey);

  if (typeof note === "number" && Number.isFinite(note)) {
    const step = Math.max(0, Math.round(Math.abs(note)));

    if (phrases && phrases.length) {
      if (step === 0 && state.lastStep > 0) {
        advanceMelodyVariation(baseKey, state);
      }
      state.lastStep = step;
      const phrase = phrases[state.phraseIndex % phrases.length];
      const value = phrase[step % phrase.length];
      return transposeFrequency(value, state.transposeSemitone);
    }

    return transposeFrequency(palette[step % palette.length], state.transposeSemitone);
  }

  if (phrases && phrases.length) {
    const phrase = phrases[state.phraseIndex % phrases.length];
    const value = phrase[state.freeStep % phrase.length];
    state.freeStep = (state.freeStep + 1) % phrase.length;
    if (state.freeStep === 0) {
      advanceMelodyVariation(baseKey, state);
    }
    return transposeFrequency(value, state.transposeSemitone);
  }

  const value = palette[state.freeStep % palette.length];
  state.freeStep = (state.freeStep + 1) % palette.length;
  return transposeFrequency(value, state.transposeSemitone);
}

function frequencyFromNote(note, fallback = 440) {
  if (Number.isFinite(note)) return note;

  const fallbackFreq = Number.isFinite(fallback) ? fallback : 440;
  if (typeof note !== "string") return fallbackFreq;

  const normalized = note.trim();
  if (!normalized) return fallbackFreq;

  const match = /^([A-Ga-g])([#b]?)(-?\d+)?$/.exec(normalized);
  if (!match) return fallbackFreq;

  const letter = match[1].toUpperCase();
  const accidental = match[2] || "";
  const noteKey = letter + accidental;
  const semitone = NOTE_INDEX[noteKey];
  if (!Number.isFinite(semitone)) return fallbackFreq;

  const octave = match[3] == null ? 4 : Number(match[3]);
  if (!Number.isFinite(octave)) return fallbackFreq;

  const midi = (octave + 1) * 12 + semitone;
  const frequency = 440 * Math.pow(2, (midi - 69) / 12);
  return Math.min(2600, Math.max(70, frequency));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function nowTime() {
  return audioCtx ? audioCtx.currentTime : 0;
}

function getPlaybackPan() {
  if (Number.isFinite(playbackContext?.pan)) {
    return clamp(playbackContext.pan, -1, 1);
  }
  const handKey = String(playbackContext?.handKey || "").toLowerCase();
  if (handKey === "left") return -0.72;
  if (handKey === "right") return 0.72;
  return 0;
}

function connectNode(node, {
  gainValue = 1,
  reverbSend = REVERB_SEND_DEFAULT,
  delaySend = DELAY_SEND_DEFAULT
} = {}) {
  if (!masterGain) return null;
  const gainNode = audioCtx.createGain();
  gainNode.gain.value = gainValue;
  node.connect(gainNode);

  let outputNode = gainNode;
  if (typeof audioCtx.createStereoPanner === "function") {
    const panner = audioCtx.createStereoPanner();
    panner.pan.value = getPlaybackPan();
    gainNode.connect(panner);
    outputNode = panner;
  }

  outputNode.connect(masterGain);

  if (reverbInputGain && reverbSend > 0) {
    const reverbSendGain = audioCtx.createGain();
    reverbSendGain.gain.value = clamp(reverbSend, 0, 1);
    outputNode.connect(reverbSendGain);
    reverbSendGain.connect(reverbInputGain);
  }

  if (delayInputGain && delaySend > 0) {
    const delaySendGain = audioCtx.createGain();
    delaySendGain.gain.value = clamp(delaySend, 0, 1);
    outputNode.connect(delaySendGain);
    delaySendGain.connect(delayInputGain);
  }

  return gainNode;
}

function createNoiseBuffer(duration = 0.12) {
  const length = Math.max(1, Math.floor(audioCtx.sampleRate * duration));
  const buffer = audioCtx.createBuffer(1, length, audioCtx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < length; i += 1) {
    data[i] = (Math.random() * 2 - 1) * (1 - i / length);
  }
  return buffer;
}

function createImpulseResponse(duration = 1.2, decay = 2.4) {
  const ctx = ensureAudioContext();
  if (!ctx) return null;
  const length = Math.max(1, Math.floor(ctx.sampleRate * duration));
  const impulse = ctx.createBuffer(2, length, ctx.sampleRate);

  for (let channel = 0; channel < 2; channel += 1) {
    const data = impulse.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      const t = i / length;
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, decay);
    }
  }

  return impulse;
}

function playTone(frequency, {
  type = "sine",
  attack = 0.01,
  decay = 0.18,
  sustain = 0.001,
  release = 0.2,
  gain = 0.18,
  vibrato = 0,
  filterType = "",
  filterFrequency = 1200,
  detuneCents = 0,
  reverbSend = REVERB_SEND_DEFAULT,
  delaySend = DELAY_SEND_DEFAULT
} = {}) {
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) return false;
  if (ctx.state !== "running") {
    void unlockAudioContext();
    if (ctx.state !== "running") return false;
  }

  if (activeVoiceCount >= MAX_ACTIVE_VOICES) return false;

  const t = nowTime();
  const osc = ctx.createOscillator();
  const amp = ctx.createGain();
  const outTarget = filterType ? ctx.createBiquadFilter() : null;

  osc.type = type;
  osc.frequency.setValueAtTime(frequency, t);
  if (Number.isFinite(detuneCents) && detuneCents !== 0) {
    osc.detune.setValueAtTime(detuneCents, t);
  }

  if (vibrato > 0) {
    const lfo = ctx.createOscillator();
    const lfoGain = ctx.createGain();
    lfo.frequency.value = 5.2;
    lfoGain.gain.value = vibrato;
    lfo.connect(lfoGain);
    lfoGain.connect(osc.frequency);
    lfo.start(t);
    lfo.stop(t + attack + decay + release + 0.08);
  }

  const densityScale = 1 / Math.sqrt(1 + activeVoiceCount * 0.35);
  const targetGain = Math.max(0.0001, gain * densityScale);

  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.linearRampToValueAtTime(targetGain, t + attack);
  amp.gain.exponentialRampToValueAtTime(Math.max(0.0001, sustain), t + attack + decay);
  amp.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay + release);

  osc.connect(amp);
  if (outTarget) {
    outTarget.type = filterType;
    outTarget.frequency.value = filterFrequency;
    amp.connect(outTarget);
    connectNode(outTarget, { reverbSend, delaySend });
  } else {
    connectNode(amp, { reverbSend, delaySend });
  }

  activeVoiceCount += 1;
  osc.onended = () => {
    activeVoiceCount = Math.max(0, activeVoiceCount - 1);
  };

  osc.start(t);
  osc.stop(t + attack + decay + release + 0.08);
  return true;
}

function playChord(frequencies, options = {}) {
  frequencies.forEach((frequency, index) => {
    playTone(frequencyFromNote(frequency), {
      type: options.type || "triangle",
      attack: options.attack ?? 0.01,
      decay: options.decay ?? 0.26,
      sustain: options.sustain ?? 0.001,
      release: options.release ?? 0.2,
      gain: (options.gain ?? 0.12) / Math.max(1, frequencies.length) * (index === 0 ? 1.15 : 1),
      filterType: options.filterType || "",
      filterFrequency: options.filterFrequency ?? 1600,
      vibrato: options.vibrato ?? 0,
      reverbSend: options.reverbSend ?? REVERB_SEND_DEFAULT,
      delaySend: options.delaySend ?? DELAY_SEND_DEFAULT,
      detuneCents: index === 0 ? 0 : (index % 2 === 0 ? -4 : 4)
    });
  });
}

function playFairyCompanion(baseFrequency, gain = 0.05) {
  if (!Number.isFinite(baseFrequency) || baseFrequency <= 0) return;
  if (activeVoiceCount >= COMPANION_VOICE_THRESHOLD) return;
  const octave = Math.min(baseFrequency * 2, 2200);
  const fifth = Math.min(baseFrequency * 1.5, 1800);

  playTone(octave, {
    type: "sine",
    attack: 0.02,
    decay: 0.24,
    sustain: 0.0008,
    release: 0.24,
    gain,
    vibrato: 4,
    filterType: "lowpass",
    filterFrequency: 3200,
    reverbSend: 0.28,
    delaySend: 0.1,
    detuneCents: -3
  });

  playTone(fifth, {
    type: "triangle",
    attack: 0.012,
    decay: 0.18,
    sustain: 0.0008,
    release: 0.2,
    gain: gain * 0.72,
    vibrato: 2,
    filterType: "lowpass",
    filterFrequency: 2600,
    reverbSend: 0.2,
    delaySend: 0.06,
    detuneCents: 3
  });
}

function playDrumHit(pattern = "kick") {
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) return false;
  if (ctx.state !== "running") {
    void unlockAudioContext();
    if (ctx.state !== "running") return false;
  }
  const t = nowTime();

  if (pattern === "kick") {
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(110, t);
    osc.frequency.exponentialRampToValueAtTime(42, t + 0.12);
    amp.gain.setValueAtTime(0.0001, t);
    amp.gain.exponentialRampToValueAtTime(0.42, t + 0.01);
    amp.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
    osc.connect(amp);
    connectNode(amp, { gainValue: 1, reverbSend: 0.02, delaySend: 0 });
    osc.start(t);
    osc.stop(t + 0.2);
    return true;
  }

  const src = ctx.createBufferSource();
  const filter = ctx.createBiquadFilter();
  const amp = ctx.createGain();
  src.buffer = createNoiseBuffer(pattern === "hat" ? 0.05 : 0.1);
  filter.type = pattern === "snare" ? "bandpass" : "highpass";
  filter.frequency.value = pattern === "snare" ? 1800 : 5200;
  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.exponentialRampToValueAtTime(pattern === "snare" ? 0.16 : 0.09, t + 0.006);
  amp.gain.exponentialRampToValueAtTime(0.0001, t + (pattern === "snare" ? 0.12 : 0.06));
  src.connect(filter);
  filter.connect(amp);
  connectNode(amp, { gainValue: 1, reverbSend: 0.03, delaySend: 0 });
  src.start(t);
  src.stop(t + (pattern === "snare" ? 0.12 : 0.06));
  return true;
}

export function setPlaybackContext(context) {
  playbackContext = context && typeof context === "object" ? { ...context } : null;
}

export function ensureAudioContext() {
  if (audioCtx) return audioCtx;
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) return null;

  audioCtx = new AudioContextCtor();
  masterGain = audioCtx.createGain();
  ambientGain = audioCtx.createGain();
  masterLimiter = audioCtx.createDynamicsCompressor();

  masterLimiter.threshold.value = -16;
  masterLimiter.knee.value = 12;
  masterLimiter.ratio.value = 2.8;
  masterLimiter.attack.value = 0.004;
  masterLimiter.release.value = 0.18;

  masterGain.gain.value = soundEnabled ? MASTER_TARGET_GAIN : 0;
  ambientGain.gain.value = 0;

  reverbInputGain = audioCtx.createGain();
  reverbWetGain = audioCtx.createGain();
  reverbNode = audioCtx.createConvolver();

  reverbInputGain.gain.value = 0.38;
  reverbWetGain.gain.value = 0.16;
  reverbNode.buffer = createImpulseResponse(1.25, 2.5);

  delayInputGain = audioCtx.createGain();
  delayNode = audioCtx.createDelay(0.5);
  delayFeedbackGain = audioCtx.createGain();
  delayToneFilter = audioCtx.createBiquadFilter();
  delayWetGain = audioCtx.createGain();

  delayInputGain.gain.value = 0.3;
  delayNode.delayTime.value = 0.17;
  delayFeedbackGain.gain.value = 0.18;
  delayToneFilter.type = "lowpass";
  delayToneFilter.frequency.value = 2200;
  delayWetGain.gain.value = 0.1;

  ambientGain.connect(masterGain);

  reverbInputGain.connect(reverbNode);
  reverbNode.connect(reverbWetGain);
  reverbWetGain.connect(masterGain);

  delayInputGain.connect(delayNode);
  delayNode.connect(delayToneFilter);
  delayToneFilter.connect(delayWetGain);
  delayWetGain.connect(masterGain);
  delayToneFilter.connect(delayFeedbackGain);
  delayFeedbackGain.connect(delayNode);

  masterGain.connect(masterLimiter);
  masterLimiter.connect(audioCtx.destination);
  return audioCtx;
}

export async function unlockAudioContext() {
  const ctx = ensureAudioContext();
  if (!ctx) return false;
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
  if (ctx.state === "interrupted") {
    try {
      await ctx.resume();
    } catch {
      // ignore and report final state below
    }
  }
  unlocked = ctx.state === "running";
  return unlocked;
}

export function toggleSound() {
  soundEnabled = !soundEnabled;
  if (masterGain) {
    masterGain.gain.setTargetAtTime(soundEnabled ? MASTER_TARGET_GAIN : 0, nowTime(), 0.03);
  }
  if (!soundEnabled) stopAllMelodies();
  return getAudioState();
}

export function getAudioState() {
  return {
    ready: Boolean(audioCtx),
    unlocked,
    running: Boolean(soundEnabled && audioCtx && audioCtx.state === "running")
  };
}

export function startAmbientLoop() {
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) return;
  if (ambientGain) {
    ambientGain.gain.setTargetAtTime(0.028, nowTime(), 0.6);
  }
}

export function stopAmbientLoop() {
  if (ambientGain) {
    ambientGain.gain.setTargetAtTime(0, nowTime(), 0.3);
  }
}

export function playDrumMushroom() {
  const ok = playDrumHit("kick");
  if (ok) emitSoundPlayed("drum-mushroom", "synth");
}

export function playXylophoneVine() {
  const ok = playTone(frequencyFromNote(chooseFromPalette("xylophone")), {
    type: "triangle",
    attack: 0.005,
    decay: 0.3,
    gain: 0.16,
    filterType: "lowpass",
    filterFrequency: 2400
  });
  if (ok) emitSoundPlayed("xylophone-vine", "synth");
}

export function playTambourineFlower() {
  const chord = chooseFromPalette("tambourine");
  playChord(Array.isArray(chord) ? chord : [chord], {
    type: "triangle",
    attack: 0.01,
    decay: 0.24,
    gain: 0.2,
    filterType: "lowpass",
    filterFrequency: 1800
  });
  emitSoundPlayed("tambourine-flower", "synth");
}

export function playFistBeat() {
  const ok = playDrumHit("kick");
  if (ok) emitSoundPlayed("fist", "synth");
}

export function playPinkyChime() {
  playChord([1046.5, 1318.51], {
    type: "sine",
    attack: 0.005,
    decay: 0.34,
    gain: 0.1,
    vibrato: 2
  });
  emitSoundPlayed("pinky", "synth");
}

export function playHeartBloom() {
  playTone(frequencyFromNote(chooseFromPalette("whistle")), {
    type: "sine",
    attack: 0.02,
    decay: 0.42,
    gain: 0.12,
    vibrato: 6
  });
  emitSoundPlayed("heart", "synth");
}

export function playAnimalRoll() {
  const palette = melodyPalettes.animal;
  const a = palette[Math.floor(Math.random() * palette.length)];
  const b = palette[Math.floor(Math.random() * palette.length)];
  playTone(a, { type: "triangle", attack: 0.01, decay: 0.18, gain: 0.12 });
  window.setTimeout(() => {
    playTone(b, { type: "triangle", attack: 0.01, decay: 0.18, gain: 0.1 });
  }, 80);
  emitSoundPlayed("animal", "synth");
}

export function playKids_Drum(note) {
  const target = chooseFromPalette("drum", note);
  const freq = frequencyFromNote(target);
  const ok = playTone(freq, {
    type: "triangle",
    attack: 0.004,
    decay: 0.16,
    release: 0.14,
    gain: 0.18,
    filterType: "lowpass",
    filterFrequency: 900,
    reverbSend: 0.04,
    delaySend: 0.01
  });
  playDrumHit("kick");
  if (ok) emitSoundPlayed("kids-drum", "synth");
}

export function playKids_Piano(note) {
  const freq = frequencyFromNote(chooseFromPalette("piano", note), 261.63);
  const ok = playTone(freq, {
    type: "triangle",
    attack: 0.006,
    decay: 0.26,
    release: 0.2,
    gain: 0.13,
    filterType: "lowpass",
    filterFrequency: 2300,
    reverbSend: 0.18,
    delaySend: 0.06
  });
  playTone(freq * 2, {
    type: "sine",
    attack: 0.01,
    decay: 0.22,
    release: 0.16,
    gain: 0.028,
    filterType: "lowpass",
    filterFrequency: 3200,
    reverbSend: 0.2,
    delaySend: 0.05
  });
  if (ok) emitSoundPlayed("kids-piano", "synth");
}

export function playKids_Guitar(note) {
  const root = frequencyFromNote(chooseFromPalette("guitar", note), 196);
  const chord = [root, root * 1.25, root * 1.5].map((f) => Math.min(f, 1760));
  playChord(chord, {
    type: "triangle",
    attack: 0.004,
    decay: 0.2,
    release: 0.16,
    gain: 0.14,
    filterType: "bandpass",
    filterFrequency: 1400,
    reverbSend: 0.16,
    delaySend: 0.05
  });
  emitSoundPlayed("kids-guitar", "synth");
}

export function playKids_Flute(note) {
  const freq = frequencyFromNote(chooseFromPalette("flute", note), 523.25);
  const ok = playTone(freq, {
    type: "sine",
    attack: 0.02,
    decay: 0.34,
    release: 0.3,
    gain: 0.1,
    vibrato: 5,
    filterType: "lowpass",
    filterFrequency: 3000,
    reverbSend: 0.3,
    delaySend: 0.12
  });
  if (ok) emitSoundPlayed("kids-flute", "synth");
}

export function playKids_Violin(note) {
  const freq = frequencyFromNote(chooseFromPalette("violin", note), 392);
  const ok = playTone(freq, {
    type: "sawtooth",
    attack: 0.02,
    decay: 0.3,
    release: 0.28,
    gain: 0.09,
    vibrato: 2.8,
    filterType: "lowpass",
    filterFrequency: 2200,
    reverbSend: 0.26,
    delaySend: 0.1
  });
  if (ok) {
    playFairyCompanion(freq, 0.024);
    emitSoundPlayed("kids-violin", "synth");
  }
}

export function playKids_Bell(note) {
  const freq = frequencyFromNote(chooseFromPalette("bell", note), 987.77);
  const ok = playTone(freq, {
    type: "triangle",
    attack: 0.004,
    decay: 0.2,
    release: 0.24,
    gain: 0.05,
    filterType: "highpass",
    filterFrequency: 2400,
    reverbSend: 0.28,
    delaySend: 0.14
  });
  playTone(freq * 2, {
    type: "sine",
    attack: 0.002,
    decay: 0.18,
    release: 0.2,
    gain: 0.018,
    filterType: "highpass",
    filterFrequency: 3200,
    reverbSend: 0.3,
    delaySend: 0.16
  });
  if (ok) emitSoundPlayed("kids-bell", "synth");
}

// 하위 호환: 기존 키를 새 악기군으로 매핑
export function playKids_Tambourine(note) {
  playKids_Guitar(note);
}

export function playKids_Xylophone(note) {
  playKids_Piano(note);
}

export function playKids_Whistle(note) {
  playKids_Flute(note);
}

export function playKids_Triangle(note) {
  playKids_Bell(note);
}

export function playKids_AnimalSurprise() {
  if (Math.random() < 0.5) {
    playKids_Violin();
    return;
  }
  playKids_Bell();
}

export function startMelodySequence(instrumentType, playFunction) {
  if (!instrumentType || typeof playFunction !== "function") return;
  stopMelodySequence(instrumentType);

  const [rawMelodyName, rawHandKey] = String(instrumentType).split(":");
  const melodyName = normalizeMelodyName(rawMelodyName);
  const handKey = rawHandKey || "default";
  const phrases = melodyPhrases[melodyName];

  if (phrases && phrases.length > 0) {
    const state = getMelodyState(melodyName, handKey);
    const phraseChoices = phrases.map((_, index) => index);
    state.phraseIndex = pickDifferentOption(phraseChoices, state.phraseIndex);

    const transposeChoices = melodyTranspositions[melodyName] || [0];
    state.transposeSemitone = pickDifferentOption(transposeChoices, state.transposeSemitone);
    state.lastStep = -1;
    state.freeStep = 0;
  }

  let step = Math.floor(Math.random() * 4);
  const tick = () => {
    if (!soundEnabled || !audioCtx || audioCtx.state !== "running") return;

    // 오디오 노드가 과밀하면 한 박자 쉬어 CPU/오디오 과부하를 완화
    if (activeVoiceCount >= ADAPTIVE_MELODY_SLOWDOWN_THRESHOLD && step % 2 === 1) {
      step = (step + 1) % 16;
      return;
    }

    playFunction(step);
    step = (step + 1) % 16;
  };

  tick();
  const timer = window.setInterval(tick, melodyTickMs);
  activeMelodies.set(instrumentType, { timer });
}

export function stopMelodySequence(instrumentType) {
  const token = activeMelodies.get(instrumentType);
  if (token && typeof token === "object" && Number.isFinite(token.timer)) {
    window.clearInterval(token.timer);
  }
  if (Number.isFinite(token)) {
    window.clearInterval(token);
  }
  activeMelodies.delete(instrumentType);
}

export function stopAllMelodies() {
  Array.from(activeMelodies.keys()).forEach((key) => {
    if (String(key).startsWith("step:")) return;
    stopMelodySequence(key);
  });
}

export function isMelodyPlaying(instrumentType) {
  return activeMelodies.has(instrumentType);
}
