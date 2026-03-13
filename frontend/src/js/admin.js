// [admin.js] 악기 식물들의 위치를 내 맘대로 옮길 수 있는 '설정 페이지'용 파일입니다.
// 마우스로 드래그해서 위치를 바꾸고 저장하면, 나중에 게임을 켤 때 그 위치에 악기가 나타납니다.

const SOUND_MAPPING_KEY = "jamjam.soundMapping.v1";
const CUSTOM_SOUNDS_KEY = "jamjam.customSounds.v1"; // 사용자가 업로드한 커스텀 사운드를 저장하는 키입니다.

const defaultSoundMapping = {
  drum: "drum",
  xylophone: "xylophone",
  tambourine: "tambourine",
  fern: "pinky",
  owl: "heart"
};

const soundOptions = [
  { value: "drum", label: "드럼" },
  { value: "xylophone", label: "피리" },
  { value: "tambourine", label: "피아노" },
  { value: "pinky", label: "심벌즈" },
  { value: "heart", label: "고양이/랜덤" },
  { value: "animal", label: "애니멀" },
  { value: "fist", label: "타격" }
];

const soundInputs = {
  drum: document.getElementById("sound-drum"),
  xylophone: document.getElementById("sound-xylophone"),
  tambourine: document.getElementById("sound-tambourine"),
  fern: document.getElementById("sound-fern"),
  owl: document.getElementById("sound-owl")
};

const uploadInputs = {
  drum: document.getElementById("upload-drum"),
  xylophone: document.getElementById("upload-xylophone"),
  tambourine: document.getElementById("upload-tambourine"),
  fern: document.getElementById("upload-fern"),
  owl: document.getElementById("upload-owl")
};

const testButtons = {
  drum: document.getElementById("test-drum"),
  xylophone: document.getElementById("test-xylophone"),
  tambourine: document.getElementById("test-tambourine"),
  fern: document.getElementById("test-fern"),
  owl: document.getElementById("test-owl")
};

const saveButton = document.getElementById("saveLayout"); // 저장 버튼입니다.
const resetButton = document.getElementById("resetLayout"); // 초기화 버튼입니다.

let audioContext = null; // 오디오 재생을 위한 컨텍스트입니다.
let customSounds = {}; // 업로드된 커스텀 사운드들을 저장합니다 (objectURL 형태)

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
      const candidate = String(parsed?.[id] || "");
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

// MP3 파일을 업로드하여 Base64로 저장하는 기능입니다.
function handleFileUpload(objectId, file) {
  if (!file || !file.type.match(/audio\/(mp3|mpeg)/)) {
    alert("MP3 파일만 업로드 가능합니다.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const base64Data = e.target.result;
    // localStorage에 저장
    const customSoundsData = loadCustomSounds();
    customSoundsData[objectId] = {
      name: file.name,
      data: base64Data,
      timestamp: Date.now()
    };
    localStorage.setItem(CUSTOM_SOUNDS_KEY, JSON.stringify(customSoundsData));

    // 메모리에도 저장
    customSounds[objectId] = base64Data;

    // 드롭다운에 새 옵션 추가
    updateSoundOptions(objectId, file.name);

    alert(`${file.name} 업로드 완료!`);
  };
  reader.readAsDataURL(file);
}

// localStorage에서 커스텀 사운드 불러오기
function loadCustomSounds() {
  try {
    const raw = localStorage.getItem(CUSTOM_SOUNDS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed || {};
  } catch {
    return {};
  }
}

// 드롭다운 옵션에 업로드된 파일 추가
function updateSoundOptions(objectId, fileName) {
  const selectEl = soundInputs[objectId];
  if (!selectEl) return;

  const customValue = `custom_${objectId}`;

  // 기존 커스텀 옵션 제거
  const existingOption = selectEl.querySelector(`option[value="${customValue}"]`);
  if (existingOption) {
    existingOption.remove();
  }

  // 새 커스텀 옵션 추가
  const option = document.createElement("option");
  option.value = customValue;
  option.textContent = `🎵 ${fileName}`;
  selectEl.appendChild(option);
  selectEl.value = customValue;
}

// 오디오 컨텍스트 초기화
function getAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

// 사운드 테스트 재생 기능
async function testPlaySound(objectId) {
  const ctx = getAudioContext();
  const selectedSound = soundInputs[objectId]?.value;

  if (!selectedSound) {
    alert("먼저 소리를 선택해주세요.");
    return;
  }

  let audioUrl = null;

  // 커스텀 사운드인 경우
  if (selectedSound.startsWith("custom_")) {
    const customSoundsData = loadCustomSounds();
    const soundData = customSoundsData[objectId];
    if (soundData && soundData.data) {
      audioUrl = soundData.data;
    } else {
      alert("커스텀 사운드를 찾을 수 없습니다.");
      return;
    }
  } else {
    // 기본 사운드인 경우 - 실제 파일 경로 매핑
    const soundMap = {
      "drum": "/assets/sounds/드럼.mp3",
      "xylophone": "/assets/sounds/피리.mp3",
      "tambourine": "/assets/sounds/피아노.mp3",
      "pinky": "/assets/sounds/심벌즈.mp3",
      "heart": "/assets/sounds/고양이.mp3",
      "animal": "/assets/sounds/고양이.mp3",
      "fist": "/assets/sounds/드럼.mp3"
    };
    audioUrl = soundMap[selectedSound] || soundMap["drum"];
  }

  try {
    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start(0);

    console.log(`Testing sound for ${objectId}: ${selectedSound}`);
  } catch (error) {
    console.error("Sound playback error:", error);
    alert("소리 재생 중 오류가 발생했습니다.");
  }
}

function bind() {
  // 파일 업로드 이벤트 연결
  Object.keys(uploadInputs).forEach((id) => {
    const uploadInput = uploadInputs[id];
    if (!uploadInput) return;

    uploadInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFileUpload(id, file);
      }
    });
  });

  // 사운드 테스트 버튼 이벤트 연결
  Object.keys(testButtons).forEach((id) => {
    const testBtn = testButtons[id];
    if (!testBtn) return;

    testBtn.addEventListener("click", () => {
      testPlaySound(id);
    });
  });

  saveButton.addEventListener("click", () => {
    const soundMap = collectSoundMapping();
    localStorage.setItem(SOUND_MAPPING_KEY, JSON.stringify(soundMap));
    saveButton.textContent = "저장됨"; // 버튼 글자를 잠깐 바꿔 저장 완료를 알려줍니다.
    window.setTimeout(() => {
      saveButton.textContent = "저장"; // 잠시 후 원래 글자로 되돌립니다.
    }, 900);
  });

  resetButton.addEventListener("click", () => {
    localStorage.removeItem(SOUND_MAPPING_KEY);
    localStorage.removeItem(CUSTOM_SOUNDS_KEY); // 커스텀 사운드도 삭제합니다.
    const soundMap = { ...defaultSoundMapping };
    populateSoundOptions(); // 사운드 옵션을 다시 초기화합니다.
    syncSoundInputs(soundMap);
  });
}

// 페이지가 열릴 때 가장 먼저 실행되어, 저장된 소리 설정을 불러오고 화면을 구성하는 기능입니다.
function init() {
  populateSoundOptions();

  // 저장된 커스텀 사운드 불러오기
  const customSoundsData = loadCustomSounds();
  Object.keys(customSoundsData).forEach((objectId) => {
    const soundData = customSoundsData[objectId];
    if (soundData && soundData.name) {
      customSounds[objectId] = soundData.data;
      updateSoundOptions(objectId, soundData.name);
    }
  });

  const soundMap = loadSoundMapping();
  syncSoundInputs(soundMap);
  bind(); // 마지막으로 버튼과 입력칸 이벤트를 연결합니다.
}

init(); // 파일이 열리자마자 관리자 페이지를 실제로 준비합니다.
