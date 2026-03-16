// [sound_mapping.js] 악기별 소리 프로필 선택과 localStorage 저장값 병합을 담당하는 모듈입니다.

// localStorage 에 저장할 때 사용할 이름표(key)입니다.
export const SOUND_MAPPING_KEY = "jamjam.soundMapping.v1";
export const CUSTOM_SOUNDS_KEY = "jamjam.customSounds.v1";
export const GESTURE_MAPPING_KEY = "jamjam.gestureMapping.v1";

// 사용자가 별도 설정을 안 했을 때의 기본 사운드 연결표입니다.
export const DEFAULT_SOUND_MAPPING = {
  drum: "drum",
  xylophone: "xylophone",
  tambourine: "tambourine",
  a: "pinky"
};

export const DEFAULT_GESTURE_MAPPING = {
  Fist: "drum",
  OpenPalm: "xylophone",
  V: "tambourine",
  Pinky: "a",
  Animal: "a",
  KHeart: "a"
};

const LEGACY_INSTRUMENT_IDS = {
  a: "fern"
};

// 저장된 사운드 매핑을 읽어오되, 이상한 값은 버리고 안전한 기본값과 섞어줍니다.
export function loadCustomSounds() {
  try {
    const raw = localStorage.getItem(CUSTOM_SOUNDS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function loadSoundMapping(soundProfiles, customSounds = loadCustomSounds()) {
  try {
    const raw = localStorage.getItem(SOUND_MAPPING_KEY);
    if (!raw) return { ...DEFAULT_SOUND_MAPPING };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_SOUND_MAPPING };

    const merged = { ...DEFAULT_SOUND_MAPPING };
    Object.keys(merged).forEach((instrumentId) => {
      const legacyId = LEGACY_INSTRUMENT_IDS[instrumentId];
      const candidate = String(parsed?.[instrumentId] || parsed?.[legacyId] || parsed?.a2 || parsed?.owl || "");
      if (candidate in soundProfiles) {
        merged[instrumentId] = candidate;
        return;
      }
      const customKey = `custom_${instrumentId}`;
      if (candidate === customKey && customSounds?.[instrumentId]?.data) {
        merged[instrumentId] = candidate;
      }
    });
    return merged;
  } catch {
    return { ...DEFAULT_SOUND_MAPPING };
  }
}

export function loadGestureMapping() {
  try {
    const raw = localStorage.getItem(GESTURE_MAPPING_KEY);
    if (!raw) return { ...DEFAULT_GESTURE_MAPPING };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_GESTURE_MAPPING };

    const merged = { ...DEFAULT_GESTURE_MAPPING };
    Object.keys(merged).forEach((gestureLabel) => {
      const candidate = String(parsed?.[gestureLabel] || "");
      if (candidate === "drum" || candidate === "xylophone" || candidate === "tambourine" || candidate === "a") {
        merged[gestureLabel] = candidate;
      }
    });
    return merged;
  } catch {
    return { ...DEFAULT_GESTURE_MAPPING };
  }
}

// 특정 악기가 현재 어떤 사운드 프로필을 써야 하는지 최종 결정을 내려줍니다.
export function getSoundProfileForInstrument(soundMapping, defaultMapping, soundProfiles, instrumentId) {
  const profileKey = soundMapping?.[instrumentId] || defaultMapping[instrumentId] || "drum";
  return soundProfiles[profileKey] || soundProfiles.drum;
}
