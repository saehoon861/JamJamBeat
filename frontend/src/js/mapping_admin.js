import {
  SOUND_MAPPING_KEY,
  CUSTOM_SOUNDS_KEY,
  GESTURE_MAPPING_KEY,
  DEFAULT_SOUND_MAPPING,
  DEFAULT_GESTURE_MAPPING
} from "./sound_mapping.js";

const soundOptions = [
  { value: "drum", label: "드럼" },
  { value: "xylophone", label: "피리" },
  { value: "tambourine", label: "피아노" },
  { value: "pinky", label: "심벌즈" },
  { value: "heart", label: "고양이/랜덤" },
  { value: "animal", label: "애니멀" },
  { value: "fist", label: "타격" }
];

const objectOptions = [
  { value: "drum", label: "고슴도치 드럼" },
  { value: "xylophone", label: "노래하는 백합" },
  { value: "tambourine", label: "행운 클로버" },
  { value: "a", label: "a 오브젝트" }
];

const soundInputs = {
  drum: document.getElementById("sound-drum"),
  xylophone: document.getElementById("sound-xylophone"),
  tambourine: document.getElementById("sound-tambourine"),
  a: document.getElementById("sound-a")
};

const uploadInputs = {
  drum: document.getElementById("upload-drum"),
  xylophone: document.getElementById("upload-xylophone"),
  tambourine: document.getElementById("upload-tambourine"),
  a: document.getElementById("upload-a")
};

const testButtons = {
  drum: document.getElementById("test-drum"),
  xylophone: document.getElementById("test-xylophone"),
  tambourine: document.getElementById("test-tambourine"),
  a: document.getElementById("test-a")
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

let audioContext = null;
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
    if (soundOptions.some((option) => option.value === candidate) || candidate === `custom_${id}`) {
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

function loadCustomSounds() {
  return loadStoredJson(CUSTOM_SOUNDS_KEY) || {};
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
    if (soundOptions.some((option) => option.value === value) || value === `custom_${id}`) {
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

function updateSoundOptions(objectId, fileName) {
  const selectEl = soundInputs[objectId];
  if (!selectEl) return;

  const customValue = `custom_${objectId}`;
  const existingOption = selectEl.querySelector(`option[value="${customValue}"]`);
  if (existingOption) {
    existingOption.remove();
  }

  const option = document.createElement("option");
  option.value = customValue;
  option.textContent = `커스텀: ${fileName}`;
  selectEl.appendChild(option);
  selectEl.value = customValue;
  renderButtonsForSelect(selectEl);
  updateButtonGroupState(selectEl);
}

function handleFileUpload(objectId, file) {
  if (!file || !file.type.match(/audio\/(mp3|mpeg)/)) {
    alert("MP3 파일만 업로드 가능합니다.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const customSounds = loadCustomSounds();
    customSounds[objectId] = {
      name: file.name,
      data: event.target.result,
      timestamp: Date.now()
    };
    localStorage.setItem(CUSTOM_SOUNDS_KEY, JSON.stringify(customSounds));
    updateSoundOptions(objectId, file.name);
  };
  reader.readAsDataURL(file);
}

function getAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

async function testPlaySound(objectId) {
  const selectedSound = soundInputs[objectId]?.value;
  if (!selectedSound) return;

  let audioUrl = null;
  if (selectedSound.startsWith("custom_")) {
    const customSounds = loadCustomSounds();
    audioUrl = customSounds[objectId]?.data || null;
  } else {
    const soundUrlMap = {
      drum: "/assets/sounds/드럼.mp3",
      xylophone: "/assets/sounds/피리.mp3",
      tambourine: "/assets/sounds/피아노.mp3",
      pinky: "/assets/sounds/심벌즈.mp3",
      heart: "/assets/sounds/고양이.mp3",
      animal: "/assets/sounds/고양이.mp3",
      fist: "/assets/sounds/드럼.mp3"
    };
    audioUrl = soundUrlMap[selectedSound] || soundUrlMap.drum;
  }

  if (!audioUrl) return;

  try {
    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await getAudioContext().decodeAudioData(arrayBuffer);
    const source = getAudioContext().createBufferSource();
    source.buffer = audioBuffer;
    source.connect(getAudioContext().destination);
    source.start(0);
  } catch {
    alert("소리 테스트에 실패했습니다.");
  }
}

function hydrateCustomSoundOptions() {
  const customSounds = loadCustomSounds();
  Object.entries(customSounds).forEach(([objectId, soundData]) => {
    if (!soundData?.name) return;
    updateSoundOptions(objectId, soundData.name);
  });
}

function bind() {
  Object.entries(uploadInputs).forEach(([id, input]) => {
    if (!input) return;
    input.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) handleFileUpload(id, file);
    });
  });

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
    localStorage.removeItem(CUSTOM_SOUNDS_KEY);
    populateSoundOptions();
    populateGestureOptions();
    syncSoundInputs(DEFAULT_SOUND_MAPPING);
    syncGestureInputs(DEFAULT_GESTURE_MAPPING);
  });
}

function init() {
  populateSoundOptions();
  populateGestureOptions();
  hydrateCustomSoundOptions();
  syncSoundInputs(loadSoundMapping());
  syncGestureInputs(loadGestureMapping());
  bind();
}

init();
