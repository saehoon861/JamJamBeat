// [sound_mapping.js] 고정된 제스처/사운드 매핑을 제공하는 모듈입니다.

// 오브젝트별 기본 사운드 프로필 연결표입니다.
export const DEFAULT_SOUND_MAPPING = {
  drum: "drum",
  xylophone: "piano",
  tambourine: "guitar",
  a: "bell",
  cat: "flute",
  penguin: "guitar"
};

// 제스처별 오브젝트 고정 매핑입니다.
export const DEFAULT_GESTURE_MAPPING = {
  Fist: "penguin",
  OpenPalm: "drum",
  V: "tambourine",
  Pinky: "a",
  Animal: "cat",
  KHeart: "xylophone"
};

export function loadCustomSounds() {
  return {};
}

export function loadSoundMapping() {
  return { ...DEFAULT_SOUND_MAPPING };
}

export function loadGestureMapping() {
  return { ...DEFAULT_GESTURE_MAPPING };
}

// 특정 악기가 현재 어떤 사운드 프로필을 써야 하는지 최종 결정을 내려줍니다.
export function getSoundProfileForInstrument(soundMapping, defaultMapping, soundProfiles, instrumentId) {
  const profileKey = soundMapping?.[instrumentId] || defaultMapping[instrumentId] || "drum";
  return soundProfiles[profileKey] || soundProfiles.drum;
}
