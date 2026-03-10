const INSTRUMENT_LAYOUT_KEY = "jamjam.instrumentLayout.v1";

const defaults = {
  drum: { x: 10, y: 14 },
  xylophone: { x: 38, y: 22 },
  tambourine: { x: 68, y: 18 }
};

const stageEls = {
  drum: document.getElementById("admin-drum"),
  xylophone: document.getElementById("admin-xylophone"),
  tambourine: document.getElementById("admin-tambourine")
};

const inputs = {
  drum: { x: document.getElementById("drum-x"), y: document.getElementById("drum-y") },
  xylophone: { x: document.getElementById("xylophone-x"), y: document.getElementById("xylophone-y") },
  tambourine: { x: document.getElementById("tambourine-x"), y: document.getElementById("tambourine-y") }
};

const saveButton = document.getElementById("saveLayout");
const resetButton = document.getElementById("resetLayout");

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function loadLayout() {
  try {
    const raw = localStorage.getItem(INSTRUMENT_LAYOUT_KEY);
    if (!raw) return structuredClone(defaults);
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return structuredClone(defaults);

    return {
      drum: {
        x: clamp(Number(parsed?.drum?.x ?? defaults.drum.x), 0, 90),
        y: clamp(Number(parsed?.drum?.y ?? defaults.drum.y), 0, 80)
      },
      xylophone: {
        x: clamp(Number(parsed?.xylophone?.x ?? defaults.xylophone.x), 0, 90),
        y: clamp(Number(parsed?.xylophone?.y ?? defaults.xylophone.y), 0, 80)
      },
      tambourine: {
        x: clamp(Number(parsed?.tambourine?.x ?? defaults.tambourine.x), 0, 90),
        y: clamp(Number(parsed?.tambourine?.y ?? defaults.tambourine.y), 0, 80)
      }
    };
  } catch {
    return structuredClone(defaults);
  }
}

function paint(layout) {
  ["drum", "xylophone", "tambourine"].forEach((id) => {
    const el = stageEls[id];
    const pos = layout[id];
    if (!el || !pos) return;
    el.style.left = `${pos.x}vw`;
    el.style.bottom = `${pos.y}vh`;
  });
}

function syncInputs(layout) {
  ["drum", "xylophone", "tambourine"].forEach((id) => {
    inputs[id].x.value = String(layout[id].x);
    inputs[id].y.value = String(layout[id].y);
  });
}

function collectFromInputs() {
  return {
    drum: {
      x: clamp(Number(inputs.drum.x.value), 0, 90),
      y: clamp(Number(inputs.drum.y.value), 0, 80)
    },
    xylophone: {
      x: clamp(Number(inputs.xylophone.x.value), 0, 90),
      y: clamp(Number(inputs.xylophone.y.value), 0, 80)
    },
    tambourine: {
      x: clamp(Number(inputs.tambourine.x.value), 0, 90),
      y: clamp(Number(inputs.tambourine.y.value), 0, 80)
    }
  };
}

function bind() {
  ["drum", "xylophone", "tambourine"].forEach((id) => {
    inputs[id].x.addEventListener("input", () => {
      paint(collectFromInputs());
    });
    inputs[id].y.addEventListener("input", () => {
      paint(collectFromInputs());
    });
  });

  saveButton.addEventListener("click", () => {
    const layout = collectFromInputs();
    localStorage.setItem(INSTRUMENT_LAYOUT_KEY, JSON.stringify(layout));
    saveButton.textContent = "저장됨";
    window.setTimeout(() => {
      saveButton.textContent = "저장";
    }, 900);
  });

  resetButton.addEventListener("click", () => {
    localStorage.removeItem(INSTRUMENT_LAYOUT_KEY);
    const layout = structuredClone(defaults);
    syncInputs(layout);
    paint(layout);
  });
}

function init() {
  const layout = loadLayout();
  syncInputs(layout);
  paint(layout);
  bind();
}

init();
