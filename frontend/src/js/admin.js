import * as Audio from "./audio.js";

const SOUND_MAPPING_KEY = "jamjam.soundMapping.v1";

const defaultSoundMapping = {
  drum: "drum",
  xylophone: "piano",
  tambourine: "guitar",
  a: "flute",
  cat: "violin",
  penguin: "bell"
};

const soundOptions = [
  { value: "drum", label: "드럼" },
  { value: "piano", label: "피아노" },
  { value: "guitar", label: "기타" },
  { value: "flute", label: "플룻" },
  { value: "violin", label: "바이올린" },
  { value: "bell", label: "벨" }
];

const soundInputs = {
  drum: document.getElementById("sound-drum"),
  xylophone: document.getElementById("sound-xylophone"),
  tambourine: document.getElementById("sound-tambourine"),
  a: document.getElementById("sound-a"),
  cat: document.getElementById("sound-cat"),
  penguin: document.getElementById("sound-penguin")
};

const testButtons = {
  drum: document.getElementById("test-drum"),
  xylophone: document.getElementById("test-xylophone"),
  tambourine: document.getElementById("test-tambourine"),
  a: document.getElementById("test-a"),
  cat: document.getElementById("test-cat"),
  penguin: document.getElementById("test-penguin")
};

const LEGACY_OBJECT_IDS = {
  a: "fern"
};

const saveButton = document.getElementById("saveLayout");
const resetButton = document.getElementById("resetLayout");

function populateSoundOptions() {
  Object.values(soundInputs).forEach((selectEl) => {
    if (!selectEl) return;
    selectEl.innerHTML = "";
    soundOptions.forEach((option) => {
      const el = document.createElement("option");
      el.value = option.value;
      el.textContent = option.label;
      selectEl.appendChild(el);
    });
  });
}

function loadSoundMapping() {
  try {
    const raw = localStorage.getItem(SOUND_MAPPING_KEY);
    if (!raw) return { ...defaultSoundMapping };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...defaultSoundMapping };

    const allowed = new Set(soundOptions.map((opt) => opt.value));
    const merged = { ...defaultSoundMapping };
    Object.keys(merged).forEach((id) => {
      const legacyId = LEGACY_OBJECT_IDS[id];
      const candidate = String(parsed?.[id] || parsed?.[legacyId] || parsed?.a2 || parsed?.owl || "");
      if (allowed.has(candidate)) merged[id] = candidate;
    });
    return merged;
  } catch {
    return { ...defaultSoundMapping };
  }
}

function syncSoundInputs(soundMap) {
  Object.keys(soundInputs).forEach((id) => {
    if (!soundInputs[id]) return;
    soundInputs[id].value = soundMap[id] || defaultSoundMapping[id];
  });
}

function collectSoundMapping() {
  const result = { ...defaultSoundMapping };
  Object.keys(soundInputs).forEach((id) => {
    const value = String(soundInputs[id]?.value || "");
    if (soundOptions.some((opt) => opt.value === value)) result[id] = value;
  });
  return result;
}

async function testPlaySound(objectId) {
  const selectedSound = soundInputs[objectId]?.value;
  if (!selectedSound) return;

  try {
    await Audio.unlockAudioContext();
    const soundMap = {
      drum: () => Audio.playKids_Drum(),
      piano: () => Audio.playKids_Piano(),
      guitar: () => Audio.playKids_Guitar(),
      flute: () => Audio.playKids_Flute(),
      violin: () => Audio.playKids_Violin(),
      bell: () => Audio.playKids_Bell()
    };
    soundMap[selectedSound]?.();
  } catch (error) {
    console.error("Sound playback error:", error);
    alert("소리 재생 중 오류가 발생했습니다.");
  }
}

function bind() {
  Object.keys(testButtons).forEach((id) => {
    const testBtn = testButtons[id];
    if (!testBtn) return;
    testBtn.addEventListener("click", () => {
      testPlaySound(id);
    });
  });

  saveButton.addEventListener("click", () => {
    localStorage.setItem(SOUND_MAPPING_KEY, JSON.stringify(collectSoundMapping()));
    saveButton.textContent = "저장됨";
    window.setTimeout(() => {
      saveButton.textContent = "저장";
    }, 900);
  });

  resetButton.addEventListener("click", () => {
    localStorage.removeItem(SOUND_MAPPING_KEY);
    populateSoundOptions();
    syncSoundInputs({ ...defaultSoundMapping });
  });
}

function init() {
  populateSoundOptions();
  syncSoundInputs(loadSoundMapping());
  bind();
}

init();
