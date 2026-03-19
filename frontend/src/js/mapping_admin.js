import * as Audio from "./audio.js";
import {
  SOUND_MAPPING_KEY,
  GESTURE_MAPPING_KEY,
  DEFAULT_SOUND_MAPPING,
  DEFAULT_GESTURE_MAPPING
} from "./sound_mapping.js";

const soundOptions = [
  { value: "drum", label: "드럼" },
  { value: "piano", label: "피아노" },
  { value: "guitar", label: "기타" },
  { value: "flute", label: "플룻" },
  { value: "violin", label: "바이올린" },
  { value: "bell", label: "벨" }
];

const objectOptions = [
  { value: "drum", label: "고슴도치 드럼" },
  { value: "xylophone", label: "아기 사슴" },
  { value: "tambourine", label: "아기 토끼" },
  { value: "a", label: "다람쥐" },
  { value: "cat", label: "고양이" },
  { value: "penguin", label: "팽귄" }
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

const gestureInputs = {
  Fist: document.getElementById("gesture-fist"),
  OpenPalm: document.getElementById("gesture-openpalm"),
  V: document.getElementById("gesture-v"),
  Pinky: document.getElementById("gesture-pinky"),
  Animal: document.getElementById("gesture-animal"),
  KHeart: document.getElementById("gesture-kheart")
};

const saveButton = document.getElementById("saveMappingButton");
const resetButton = document.getElementById("resetMappingButton");
const buttonGroupBySelect = new Map();

function createButtonGroup(selectEl) {
  if (!selectEl) return null;
  const group = document.createElement("div");
  group.className = "mapping-button-group";
  selectEl.classList.add("mapping-select-hidden");
  selectEl.insertAdjacentElement("afterend", group);
  buttonGroupBySelect.set(selectEl, group);
  return group;
}

function renderButtonsForSelect(selectEl) {
  if (!selectEl) return;
  const group = buttonGroupBySelect.get(selectEl) || createButtonGroup(selectEl);
  if (!group) return;

  group.innerHTML = "";
  Array.from(selectEl.options).forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mapping-option-button";
    button.textContent = option.textContent;
    button.dataset.value = option.value;
    button.classList.toggle("is-active", selectEl.value === option.value);
    button.addEventListener("click", () => {
      selectEl.value = option.value;
      updateButtonGroupState(selectEl);
    });
    group.appendChild(button);
  });
}

function updateButtonGroupState(selectEl) {
  const group = buttonGroupBySelect.get(selectEl);
  if (!group) return;
  Array.from(group.querySelectorAll(".mapping-option-button")).forEach((button) => {
    button.classList.toggle("is-active", button.dataset.value === selectEl.value);
  });
}

function loadStoredJson(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function loadSoundMapping() {
  const parsed = loadStoredJson(SOUND_MAPPING_KEY);
  const merged = { ...DEFAULT_SOUND_MAPPING };
  if (!parsed) return merged;
  Object.keys(merged).forEach((id) => {
    const candidate = String(parsed?.[id] || parsed?.fern || parsed?.owl || "");
    if (soundOptions.some((option) => option.value === candidate)) {
      merged[id] = candidate;
    }
  });
  return merged;
}

function loadGestureMapping() {
  const parsed = loadStoredJson(GESTURE_MAPPING_KEY);
  const merged = { ...DEFAULT_GESTURE_MAPPING };
  if (!parsed) return merged;
  Object.keys(merged).forEach((label) => {
    const candidate = String(parsed?.[label] || "");
    if (objectOptions.some((option) => option.value === candidate)) {
      merged[label] = candidate;
    }
  });
  return merged;
}

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
    renderButtonsForSelect(selectEl);
  });
}

function populateGestureOptions() {
  Object.values(gestureInputs).forEach((selectEl) => {
    if (!selectEl) return;
    selectEl.innerHTML = "";
    objectOptions.forEach((option) => {
      const el = document.createElement("option");
      el.value = option.value;
      el.textContent = option.label;
      selectEl.appendChild(el);
    });
    renderButtonsForSelect(selectEl);
  });
}

function syncSoundInputs(soundMap) {
  Object.entries(soundInputs).forEach(([id, selectEl]) => {
    if (!selectEl) return;
    selectEl.value = soundMap[id] || DEFAULT_SOUND_MAPPING[id];
    updateButtonGroupState(selectEl);
  });
}

function syncGestureInputs(gestureMap) {
  Object.entries(gestureInputs).forEach(([label, selectEl]) => {
    if (!selectEl) return;
    selectEl.value = gestureMap[label] || DEFAULT_GESTURE_MAPPING[label];
    updateButtonGroupState(selectEl);
  });
}

function collectSoundMapping() {
  const result = { ...DEFAULT_SOUND_MAPPING };
  Object.entries(soundInputs).forEach(([id, selectEl]) => {
    const value = String(selectEl?.value || "");
    if (soundOptions.some((option) => option.value === value)) {
      result[id] = value;
    }
  });
  return result;
}

function collectGestureMapping() {
  const result = { ...DEFAULT_GESTURE_MAPPING };
  Object.entries(gestureInputs).forEach(([label, selectEl]) => {
    const value = String(selectEl?.value || "");
    if (objectOptions.some((option) => option.value === value)) {
      result[label] = value;
    }
  });
  return result;
}

async function testPlaySound(objectId) {
  const selectedSound = soundInputs[objectId]?.value;
  if (!selectedSound) return;

  try {
    await Audio.unlockAudioContext();
    const testers = {
      drum: () => Audio.playKids_Drum(),
      piano: () => Audio.playKids_Piano(),
      guitar: () => Audio.playKids_Guitar(),
      flute: () => Audio.playKids_Flute(),
      violin: () => Audio.playKids_Violin(),
      bell: () => Audio.playKids_Bell()
    };
    testers[selectedSound]?.();
  } catch {
    alert("소리 테스트에 실패했습니다.");
  }
}

function bind() {
  Object.entries(testButtons).forEach(([id, button]) => {
    if (!button) return;
    button.addEventListener("click", () => {
      testPlaySound(id);
    });
  });

  saveButton.addEventListener("click", () => {
    localStorage.setItem(SOUND_MAPPING_KEY, JSON.stringify(collectSoundMapping()));
    localStorage.setItem(GESTURE_MAPPING_KEY, JSON.stringify(collectGestureMapping()));
    saveButton.textContent = "저장됨";
    window.setTimeout(() => {
      saveButton.textContent = "저장";
    }, 900);
  });

  resetButton.addEventListener("click", () => {
    localStorage.removeItem(SOUND_MAPPING_KEY);
    localStorage.removeItem(GESTURE_MAPPING_KEY);
    populateSoundOptions();
    populateGestureOptions();
    syncSoundInputs(DEFAULT_SOUND_MAPPING);
    syncGestureInputs(DEFAULT_GESTURE_MAPPING);
  });
}

function init() {
  populateSoundOptions();
  populateGestureOptions();
  syncSoundInputs(loadSoundMapping());
  syncGestureInputs(loadGestureMapping());
  bind();
}

init();
