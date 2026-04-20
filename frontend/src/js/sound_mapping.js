// [sound_mapping.js] 고정된 제스처/사운드 매핑을 제공하는 모듈입니다.

// 오브젝트별 기본 사운드 프로필 연결표입니다.
export const DEFAULT_SOUND_MAPPING = {
  drum: "softpad",
  xylophone: "musicbox",
  tambourine: "piano",
  a: "bell",
  cat: "flute",
  penguin: "violin"
};

// 제스처별 오브젝트 고정 매핑입니다.
export const DEFAULT_GESTURE_MAPPING = {
  Fist: "penguin",
  OpenPalm: "drum",
  V: "tambourine",
  Pinky: "a",          // Squirrel Cymbal
  Animal: "xylophone", // Fox Xylophone
  KHeart: "cat"        // Cat Heart
};

const GESTURE_MAPPING_STORAGE_KEY = "jamjam.gestureMapping.v2";
const OBJECT_SAMPLE_MAPPING_STORAGE_KEY = "jamjam.objectSampleMapping.v2";

export const OBJECT_SAMPLE_OPTIONS = [
  { id: "kick", label: "킥" },
  { id: "small-drum", label: "작은북" },
  { id: "snare", label: "스네어" },
  { id: "crash", label: "심벌" },
  { id: "maracas", label: "마라카스" },
  { id: "iloveyou", label: "I love you" },
  { id: "flute-c4", label: "플룻 도" },
  { id: "flute-d4", label: "플룻 레" },
  { id: "flute-e4", label: "플룻 미" },
  { id: "flute-f4", label: "플룻 파" },
  { id: "flute-g4", label: "플룻 솔" },
  { id: "flute-a4", label: "플룻 라" },
  { id: "flute-b4", label: "플룻 시" },
  { id: "lute-c3", label: "류트 도3" },
  { id: "lute-d3", label: "류트 레3" },
  { id: "lute-e3", label: "류트 미3" },
  { id: "lute-f3", label: "류트 파3" },
  { id: "lute-g3", label: "류트 솔3" },
  { id: "lute-a3", label: "류트 라3" },
  { id: "lute-b3", label: "류트 시3" },
  { id: "lute-c4", label: "류트 도4" },
  { id: "duck", label: "꽥꽥" },
  { id: "scratch", label: "끼리릭" },
  { id: "triangle", label: "트라이앵글" }
];

export const DEFAULT_OBJECT_SAMPLE_MAPPING = {
  drum: "flute-c4",
  xylophone: "flute-d4",
  tambourine: "flute-e4",
  a: "flute-f4",
  cat: "flute-g4",
  penguin: "flute-a4"
};

export function loadCustomSounds() {
  return {};
}

export function loadSoundMapping() {
  return { ...DEFAULT_SOUND_MAPPING };
}

export function loadGestureMapping() {
  try {
    const raw = localStorage.getItem(GESTURE_MAPPING_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_GESTURE_MAPPING };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_GESTURE_MAPPING };
    return {
      ...DEFAULT_GESTURE_MAPPING,
      ...parsed
    };
  } catch {
    return { ...DEFAULT_GESTURE_MAPPING };
  }
}

function persistGestureMapping(mapping) {
  try {
    localStorage.setItem(GESTURE_MAPPING_STORAGE_KEY, JSON.stringify(mapping));
  } catch {
    // ignore storage failures
  }
}

function dispatchGestureMappingChanged(mapping) {
  window.dispatchEvent(new CustomEvent("jamjam:gesture-mapping-changed", {
    detail: {
      mapping: { ...mapping }
    }
  }));
}

export function setGestureMapping(gestureLabel, instrumentId) {
  if (!gestureLabel || !instrumentId) return loadGestureMapping();
  const nextMapping = {
    ...loadGestureMapping(),
    [gestureLabel]: instrumentId
  };
  persistGestureMapping(nextMapping);
  dispatchGestureMappingChanged(nextMapping);
  return nextMapping;
}

export function resetGestureMapping() {
  try {
    localStorage.removeItem(GESTURE_MAPPING_STORAGE_KEY);
  } catch {
    // ignore storage failures
  }
  const nextMapping = { ...DEFAULT_GESTURE_MAPPING };
  dispatchGestureMappingChanged(nextMapping);
  return nextMapping;
}

export function loadObjectSampleMapping() {
  try {
    const raw = localStorage.getItem(OBJECT_SAMPLE_MAPPING_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_OBJECT_SAMPLE_MAPPING };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_OBJECT_SAMPLE_MAPPING };
    return {
      ...DEFAULT_OBJECT_SAMPLE_MAPPING,
      ...parsed
    };
  } catch {
    return { ...DEFAULT_OBJECT_SAMPLE_MAPPING };
  }
}

function persistObjectSampleMapping(mapping) {
  try {
    localStorage.setItem(OBJECT_SAMPLE_MAPPING_STORAGE_KEY, JSON.stringify(mapping));
  } catch {
    // ignore storage failures
  }
}

function dispatchObjectSampleMappingChanged(mapping) {
  window.dispatchEvent(new CustomEvent("jamjam:object-sample-mapping-changed", {
    detail: {
      mapping: { ...mapping }
    }
  }));
}

export function setObjectSampleMapping(instrumentId, sampleId) {
  if (!instrumentId || !sampleId) return loadObjectSampleMapping();
  const allowedIds = new Set(OBJECT_SAMPLE_OPTIONS.map((option) => option.id));
  if (!allowedIds.has(sampleId)) return loadObjectSampleMapping();
  const nextMapping = {
    ...loadObjectSampleMapping(),
    [instrumentId]: sampleId
  };
  persistObjectSampleMapping(nextMapping);
  dispatchObjectSampleMappingChanged(nextMapping);
  return nextMapping;
}

export function resetObjectSampleMapping() {
  try {
    localStorage.removeItem(OBJECT_SAMPLE_MAPPING_STORAGE_KEY);
  } catch {
    // ignore storage failures
  }
  const nextMapping = { ...DEFAULT_OBJECT_SAMPLE_MAPPING };
  dispatchObjectSampleMappingChanged(nextMapping);
  return nextMapping;
}

// 특정 악기가 현재 어떤 사운드 프로필을 써야 하는지 최종 결정을 내려줍니다.
export function getSoundProfileForInstrument(soundMapping, defaultMapping, soundProfiles, instrumentId) {
  const profileKey = soundMapping?.[instrumentId] || defaultMapping[instrumentId] || "drum";
  return soundProfiles[profileKey] || soundProfiles.drum;
}
