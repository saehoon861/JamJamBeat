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
let panDebugMode = "context";
let lastPlaybackMeta = null;
let lastOutputRouteMeta = {
  pan: 0,
  gainMode: "profile"
};
const sampleBufferByKey = new Map();
const sampleLoadPromiseByKey = new Map();
const SOUND_DEBUG = (() => {
  const raw = new URLSearchParams(window.location.search).get("soundDebug");
  return raw === "1" || raw === "true";
})();
const SOUND_DEBUG_LOG_LIMIT = 200;

function pushSoundDebug(stage, payload = {}) {
  if (!SOUND_DEBUG) return;
  const entry = {
    at: Date.now(),
    perfNow: Number(performance.now().toFixed(2)),
    stage,
    ...payload
  };
  try {
    if (!Array.isArray(window.__jamjamSoundDebugLog)) {
      window.__jamjamSoundDebugLog = [];
    }
    window.__jamjamSoundDebugLog.push(entry);
    if (window.__jamjamSoundDebugLog.length > SOUND_DEBUG_LOG_LIMIT) {
      window.__jamjamSoundDebugLog.splice(0, window.__jamjamSoundDebugLog.length - SOUND_DEBUG_LOG_LIMIT);
    }
  } catch {
    // ignore debug buffer failures
  }
  console.info("[SoundDebug]", entry);
}

const activeMelodies = new Map();
const melodyTickMs = 300;
const MELODY_STEPS = 16;
const MASTER_TARGET_GAIN = 0.82;
const REVERB_SEND_DEFAULT = 0.12;
const DELAY_SEND_DEFAULT = 0.04;
const MAX_ACTIVE_VOICES = 24;
const COMPANION_VOICE_THRESHOLD = 14;
const ADAPTIVE_MELODY_SLOWDOWN_THRESHOLD = 16;

const SAMPLE_LIBRARY = {
  drum: {
    paths: ["/assets/sounds/드럼.wav", "/assets/sounds/드럼.mp3"],
    baseFrequency: 110,
    gainValue: 0.9,
    reverbSend: 0.05,
    delaySend: 0,
    minRate: 0.78,
    maxRate: 1.28
  },
  piano: {
    paths: ["/assets/sounds/피아노.mp3"],
    baseFrequency: 261.63,
    gainValue: 0.72,
    reverbSend: 0.16,
    delaySend: 0.05,
    minRate: 0.72,
    maxRate: 1.75
  },
  guitar: {
    paths: ["/assets/sounds/기타.wav"],
    baseFrequency: 196,
    gainValue: 0.8,
    reverbSend: 0.1,
    delaySend: 0.05,
    minRate: 0.72,
    maxRate: 1.5
  },
  flute: {
    paths: ["/assets/sounds/피리.mp3"],
    baseFrequency: 523.25,
    gainValue: 0.68,
    reverbSend: 0.24,
    delaySend: 0.08,
    minRate: 0.78,
    maxRate: 1.7
  },
  violin: {
    paths: ["/assets/sounds/고양이.mp3"],
    baseFrequency: 392,
    gainValue: 0.78,
    reverbSend: 0.22,
    delaySend: 0.08,
    minRate: 0.82,
    maxRate: 1.4
  },
  bell: {
    paths: ["/assets/sounds/심벌즈.mp3"],
    baseFrequency: 987.77,
    gainValue: 0.62,
    reverbSend: 0.24,
    delaySend: 0.1,
    minRate: 0.84,
    maxRate: 1.65
  }
};

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
  bell: [783.99, 987.77, 1174.66, 1318.51, 1567.98]
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

const globalSequencer = {
  timer: null,
  currentStep: -1,
  startedAtMs: 0
};

const melodyCursor = new Map();

function normalizeMelodyName(name) {
  return name;
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

async function loadSampleBuffer(sampleKey) {
  if (sampleBufferByKey.has(sampleKey)) return sampleBufferByKey.get(sampleKey);
  if (!audioCtx) return null;

  const existingPromise = sampleLoadPromiseByKey.get(sampleKey);
  if (existingPromise) return existingPromise;

  const sampleConfig = SAMPLE_LIBRARY[sampleKey];
  if (!sampleConfig) return null;

  const loadPromise = (async () => {
    let lastError = null;
    for (const path of sampleConfig.paths) {
      try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const arrayBuffer = await response.arrayBuffer();
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        sampleBufferByKey.set(sampleKey, decoded);
        return decoded;
      } catch (error) {
        lastError = error;
      }
    }

    console.error(`[Audio] failed to load sample: ${sampleKey}`, lastError);
    return null;
  })();

  sampleLoadPromiseByKey.set(sampleKey, loadPromise);
  try {
    return await loadPromise;
  } finally {
    sampleLoadPromiseByKey.delete(sampleKey);
  }
}

function primeInstrumentSamples() {
  Object.keys(SAMPLE_LIBRARY).forEach((sampleKey) => {
    void loadSampleBuffer(sampleKey);
  });
}

async function ensureInstrumentSamplesReady() {
  await Promise.allSettled(
    Object.keys(SAMPLE_LIBRARY).map((sampleKey) => loadSampleBuffer(sampleKey))
  );
}

function finishSamplePlayback(source) {
  activeVoiceCount = Math.max(0, activeVoiceCount - 1);
  source.onended = null;
}

function playSample(sampleKey, {
  targetFrequency = null,
  gainValue,
  reverbSend,
  delaySend,
  detuneCents = 0
} = {}) {
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) {
    pushSoundDebug("audio.playSample.skip", {
      sampleKey,
      reason: !ctx ? "no_audio_context" : "sound_disabled"
    });
    return false;
  }
  if (ctx.state !== "running") {
    void unlockAudioContext();
    if (ctx.state !== "running") {
      pushSoundDebug("audio.playSample.skip", {
        sampleKey,
        reason: "context_not_running_after_unlock",
        state: ctx.state
      });
      return false;
    }
  }
  if (activeVoiceCount >= MAX_ACTIVE_VOICES) {
    pushSoundDebug("audio.playSample.skip", {
      sampleKey,
      reason: "max_active_voices",
      activeVoiceCount
    });
    return false;
  }

  const buffer = sampleBufferByKey.get(sampleKey);
  const sampleConfig = SAMPLE_LIBRARY[sampleKey];
  if (!buffer || !sampleConfig) {
    void loadSampleBuffer(sampleKey);
    pushSoundDebug("audio.playSample.skip", {
      sampleKey,
      reason: !sampleConfig ? "missing_sample_config" : "sample_buffer_not_ready"
    });
    return false;
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = Number.isFinite(targetFrequency)
    ? clamp(
      targetFrequency / sampleConfig.baseFrequency,
      sampleConfig.minRate,
      sampleConfig.maxRate
    )
    : 1;
  source.detune.value = detuneCents;

  connectNode(source, {
    gainValue: gainValue ?? sampleConfig.gainValue,
    reverbSend: reverbSend ?? sampleConfig.reverbSend,
    delaySend: delaySend ?? sampleConfig.delaySend
  });

  activeVoiceCount += 1;
  source.onended = () => finishSamplePlayback(source);
  source.start(nowTime());
  pushSoundDebug("audio.playSample.ok", {
    sampleKey,
    playbackRate: Number(source.playbackRate.value.toFixed(3)),
    activeVoiceCount,
    instrumentId: playbackContext?.instrumentId || null,
    handKey: playbackContext?.handKey || null,
    gestureLabel: playbackContext?.gestureLabel || null
  });
  return true;
}

function stopGlobalSequencer() {
  if (Number.isFinite(globalSequencer.timer)) {
    window.clearInterval(globalSequencer.timer);
  }
  globalSequencer.timer = null;
  globalSequencer.currentStep = -1;
  globalSequencer.startedAtMs = 0;
}

function tickGlobalSequencer() {
  if (!soundEnabled || !audioCtx || audioCtx.state !== "running") return;
  globalSequencer.currentStep = (globalSequencer.currentStep + 1) % MELODY_STEPS;

  activeMelodies.forEach((sequence, instrumentType) => {
    if (!sequence || typeof sequence.playFunction !== "function") return;

    if (!sequence.started) {
      if (globalSequencer.currentStep !== sequence.startOnStep) return;
      sequence.started = true;
    }

    if (activeVoiceCount >= ADAPTIVE_MELODY_SLOWDOWN_THRESHOLD && globalSequencer.currentStep % 2 === 1) {
      return;
    }

    sequence.lastStep = globalSequencer.currentStep;
    sequence.playFunction(globalSequencer.currentStep);
  });
}

function ensureGlobalSequencer() {
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) return false;
  if (ctx.state !== "running") return false;
  if (Number.isFinite(globalSequencer.timer)) return true;

  globalSequencer.currentStep = -1;
  globalSequencer.startedAtMs = performance.now();
  globalSequencer.timer = window.setInterval(tickGlobalSequencer, melodyTickMs);
  return true;
}

function getPlaybackPan(panOverride = null) {
  if (panDebugMode === "center") return 0;
  if (Number.isFinite(panOverride)) {
    return clamp(panOverride, -1, 1);
  }
  if (Number.isFinite(playbackContext?.pan)) {
    return clamp(playbackContext.pan, -1, 1);
  }
  const handKey = String(playbackContext?.handKey || "").toLowerCase();
  if (handKey === "left") return -0.28;
  if (handKey === "right") return 0.28;
  return 0;
}

function connectNode(node, {
  gainValue = 1,
  reverbSend = REVERB_SEND_DEFAULT,
  delaySend = DELAY_SEND_DEFAULT,
  pan = null,
  gainMode = "profile"
} = {}) {
  if (!masterGain) return null;
  const gainNode = audioCtx.createGain();
  gainNode.gain.value = gainValue;
  node.connect(gainNode);

  let outputNode = gainNode;
  const resolvedPan = getPlaybackPan(pan);
  if (typeof audioCtx.createStereoPanner === "function") {
    const panner = audioCtx.createStereoPanner();
    panner.pan.value = resolvedPan;
    gainNode.connect(panner);
    outputNode = panner;
  }
  lastOutputRouteMeta = {
    pan: Number(resolvedPan.toFixed(3)),
    gainMode
  };

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
  delaySend = DELAY_SEND_DEFAULT,
  pan = null,
  gainMode = "profile"
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
    connectNode(outTarget, { reverbSend, delaySend, pan, gainMode });
  } else {
    connectNode(amp, { reverbSend, delaySend, pan, gainMode });
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
    connectNode(amp, { gainValue: 1, reverbSend: 0.02, delaySend: 0, gainMode: "profile" });
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
  connectNode(amp, { gainValue: 1, reverbSend: 0.03, delaySend: 0, gainMode: "profile" });
  src.start(t);
  src.stop(t + (pattern === "snare" ? 0.12 : 0.06));
  return true;
}

export function setPlaybackContext(context) {
  playbackContext = context && typeof context === "object" ? { ...context } : null;
  pushSoundDebug("audio.setPlaybackContext", playbackContext || { context: null });
}

function captureLastPlaybackMeta(soundKey, playMode, detail = {}) {
  lastPlaybackMeta = {
    soundKey,
    playMode,
    instrumentId: detail.instrumentId ?? playbackContext?.instrumentId ?? null,
    gestureLabel: detail.gestureLabel ?? playbackContext?.gestureLabel ?? null,
    gestureSource: detail.gestureSource ?? playbackContext?.gestureSource ?? null,
    handKey: detail.handKey ?? playbackContext?.handKey ?? null,
    pan: Number.isFinite(detail.pan) ? Number(detail.pan.toFixed(3)) : lastOutputRouteMeta.pan,
    gainMode: detail.gainMode || lastOutputRouteMeta.gainMode || "profile"
  };
  return lastPlaybackMeta;
}

function emitSoundPlayed(soundKey, playMode, detail = {}) {
  const now = performance.now();
  const triggerTs = Number.isFinite(playbackContext?.triggerTs) ? playbackContext.triggerTs : null;
  const inferenceTs = Number.isFinite(playbackContext?.inferenceTs) ? playbackContext.inferenceTs : null;
  const lastPlayback = captureLastPlaybackMeta(soundKey, playMode, detail);
  const eventDetail = {
    at: Date.now(),
    soundKey,
    playMode,
    instrumentId: lastPlayback.instrumentId,
    gestureLabel: lastPlayback.gestureLabel,
    gestureSource: lastPlayback.gestureSource,
    handKey: lastPlayback.handKey,
    latencyMs: triggerTs == null ? null : (now - triggerTs),
    inferenceLatencyMs: inferenceTs == null ? null : (now - inferenceTs),
    pan: lastPlayback.pan,
    gainMode: lastPlayback.gainMode
  };

  window.dispatchEvent(new CustomEvent("jamjam:sound-played", {
    detail: eventDetail
  }));
  pushSoundDebug("audio.emitSoundPlayed", {
    soundKey,
    playMode,
    instrumentId: lastPlayback.instrumentId,
    handKey: lastPlayback.handKey,
    gestureLabel: lastPlayback.gestureLabel,
    pan: lastPlayback.pan,
    gainMode: lastPlayback.gainMode,
    triggerLatencyMs: triggerTs == null ? null : Number((now - triggerTs).toFixed(2)),
    inferenceLatencyMs: inferenceTs == null ? null : Number((now - inferenceTs).toFixed(2))
  });
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

  // 첫 입력 전에 샘플을 올려두어 합성음 fallback이 먼저 들리지 않게 합니다.
  primeInstrumentSamples();

  return audioCtx;
}

export async function unlockAudioContext() {
  const ctx = ensureAudioContext();
  if (!ctx) {
    pushSoundDebug("audio.unlock.result", {
      unlocked: false,
      reason: "no_audio_context"
    });
    return false;
  }
  pushSoundDebug("audio.unlock.start", {
    state: ctx.state,
    soundEnabled
  });
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
  if (ctx.state === "running") {
    await ensureInstrumentSamplesReady();
  }
  unlocked = ctx.state === "running";
  pushSoundDebug("audio.unlock.result", {
    unlocked,
    state: ctx.state,
    soundEnabled
  });
  return unlocked;
}

export function toggleSound() {
  soundEnabled = !soundEnabled;
  if (masterGain) {
    masterGain.gain.setTargetAtTime(soundEnabled ? MASTER_TARGET_GAIN : 0, nowTime(), 0.03);
  }
  if (!soundEnabled) stopAllMelodies();
  pushSoundDebug("audio.toggleSound", {
    soundEnabled,
    hasContext: Boolean(audioCtx),
    state: audioCtx?.state || "none"
  });
  return getAudioState();
}

export function getAudioState() {
  return {
    ready: Boolean(audioCtx),
    unlocked,
    running: Boolean(soundEnabled && audioCtx && audioCtx.state === "running")
  };
}

export function getAudioDebugState() {
  return {
    ready: Boolean(audioCtx),
    unlocked,
    running: Boolean(soundEnabled && audioCtx && audioCtx.state === "running"),
    contextState: audioCtx?.state || "none",
    soundEnabled,
    activeVoiceCount,
    panDebugMode,
    lastPlayback: lastPlaybackMeta ? { ...lastPlaybackMeta } : null
  };
}

export function setPanDebugMode(mode) {
  panDebugMode = mode === "center" ? "center" : "context";
}

export function getPanDebugMode() {
  return panDebugMode;
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

export function playKids_Drum(note) {
  const target = chooseFromPalette("drum", note);
  const freq = frequencyFromNote(target);
  const ok = playSample("drum", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-drum", "sample");
    return;
  }

  const fallback = playTone(freq, {
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
  if (fallback) emitSoundPlayed("kids-drum", "synth");
}

export function playKids_Piano(note) {
  const freq = frequencyFromNote(chooseFromPalette("piano", note), 261.63);
  const ok = playSample("piano", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-piano", "sample");
    return;
  }

  const fallback = playTone(freq, {
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
  if (fallback) emitSoundPlayed("kids-piano", "synth");
}

export function playKids_Guitar(note) {
  const freq = frequencyFromNote(chooseFromPalette("guitar", note), 196);
  const ok = playSample("guitar", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-guitar", "sample");
    return;
  }

  const chord = [freq, freq * 1.25, freq * 1.5].map((value) => Math.min(value, 1760));
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
  const ok = playSample("flute", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-flute", "sample");
    return;
  }

  const fallback = playTone(freq, {
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
  if (fallback) emitSoundPlayed("kids-flute", "synth");
}

export function playKids_Violin(note) {
  const freq = frequencyFromNote(chooseFromPalette("violin", note), 392);
  const ok = playSample("violin", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-violin", "sample");
    return;
  }

  const fallback = playTone(freq, {
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
  if (fallback) {
    playFairyCompanion(freq, 0.024);
    emitSoundPlayed("kids-violin", "synth");
  }
}

export function playKids_Bell(note) {
  const freq = frequencyFromNote(chooseFromPalette("bell", note), 987.77);
  const ok = playSample("bell", { targetFrequency: freq });
  if (ok) {
    emitSoundPlayed("kids-bell", "sample");
    return;
  }

  const fallback = playTone(freq, {
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
  if (fallback) emitSoundPlayed("kids-bell", "synth");
}

export function playKids_Triangle(note) {
  playKids_Bell(note);
}

export function playDebugBeep({
  frequency = 880,
  gain = 0.16,
  pan = 0,
  durationMs = 180
} = {}) {
  const durationSec = Math.max(0.05, durationMs / 1000);
  const ok = playTone(frequency, {
    type: "sine",
    attack: 0.004,
    decay: Math.max(0.04, durationSec * 0.5),
    sustain: 0.0001,
    release: Math.max(0.05, durationSec * 0.6),
    gain,
    filterType: "lowpass",
    filterFrequency: 2600,
    reverbSend: 0.02,
    delaySend: 0,
    pan,
    gainMode: "debug-fixed"
  });
  if (!ok) return false;

  emitSoundPlayed("debug-beep", "debug", {
    instrumentId: "debug",
    gestureLabel: "DebugBeep",
    gestureSource: "debug",
    handKey: "debug",
    pan,
    gainMode: "debug-fixed"
  });
  window.dispatchEvent(new CustomEvent("jamjam:audio-debug-played", {
    detail: {
      ...getAudioDebugState().lastPlayback,
      frequency,
      durationMs
    }
  }));
  return true;
}

export function startMelodySequence(instrumentType, playFunction) {
  if (!instrumentType || typeof playFunction !== "function") return;
  const ctx = ensureAudioContext();
  if (!ctx || !soundEnabled) return;
  if (ctx.state !== "running") {
    void unlockAudioContext();
    if (ctx.state !== "running") return;
  }

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

  const sequencerReady = ensureGlobalSequencer();
  if (!sequencerReady) return;

  activeMelodies.set(instrumentType, {
    playFunction,
    started: false,
    startOnStep: (globalSequencer.currentStep + 1 + MELODY_STEPS) % MELODY_STEPS,
    lastStep: -1
  });
}

export function stopMelodySequence(instrumentType) {
  const token = activeMelodies.get(instrumentType);
  if (!token) return;
  activeMelodies.delete(instrumentType);
  if (activeMelodies.size === 0) {
    stopGlobalSequencer();
  }
}

export function stopMelodiesForHand(handKey = "default") {
  const suffix = `:${String(handKey || "default").toLowerCase()}`;
  Array.from(activeMelodies.keys()).forEach((key) => {
    if (String(key).toLowerCase().endsWith(suffix)) {
      stopMelodySequence(key);
    }
  });
}

export function stopAllMelodies() {
  Array.from(activeMelodies.keys()).forEach((key) => {
    if (String(key).startsWith("step:")) return;
    stopMelodySequence(key);
  });
  stopGlobalSequencer();
}

export function isMelodyPlaying(instrumentType) {
  return activeMelodies.has(instrumentType);
}

export function hasAnyActiveMelody() {
  return activeMelodies.size > 0;
}
