// [particle_system.js] 악기 반응 시 튀는 파티클과 재시작형 애니메이션 효과를 맡는 모듈입니다.

// createParticleSystem(...) 은 파티클 목록을 내부에 숨긴 채,
// "추가하기(spawnBurst)" 와 "업데이트하기(updateParticles)" 만 밖으로 꺼내주는 공장 함수입니다.
export function createParticleSystem(effectCtx, effectCanvas) {
  const particles = [];

  // 특정 악기나 제스처 타입에 맞는 색과 양으로 파티클을 생성합니다.
  function spawnBurst(type, element) {
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const cx = rect.left + rect.width * 0.5;
    const cy = rect.top + rect.height * 0.5;
    const baseColors =
      type === "drum"
        ? ["#ffd88b", "#ff9f68", "#fff0b0"]
        : type === "xylophone"
          ? ["#95f5ff", "#7ff9b8", "#ffd27f", "#ff9dc2"]
          : type === "fist"
            ? ["#ff9f68", "#ffd388", "#ff7a7a"]
            : type === "openpalm"
              ? ["#9cf6ff", "#b5ffca", "#ffeab0"]
              : type === "pinky"
                ? ["#ffc6ef", "#ff9bdc", "#ffd9f4"]
                : type === "animal"
                  ? ["#b8ffa4", "#ffe695", "#8ed6ff", "#ffb5d9"]
                  : type === "kheart"
                    ? ["#ff7fcb", "#ff4fa8", "#ffd0ec", "#ffe1f4"]
                    : ["#ffc5df", "#ffe9a3", "#d0ffa8"];

    const count =
      type === "animal" ? 28
        : type === "kheart" ? 32
          : type === "fist" ? 24
            : 18;

    // count 개수만큼 파티클을 한꺼번에 만들어 배열에 넣습니다.
    for (let i = 0; i < count; i += 1) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.24;
      const speed = 1.6 + Math.random() * 3.5;
      particles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 1.2,
        gravity: 0.07,
        life: 46 + Math.random() * 34,
        maxLife: 80,
        size: type === "kheart" ? 4 + Math.random() * 5 : 3 + Math.random() * 4,
        color: baseColors[Math.floor(Math.random() * baseColors.length)]
      });
    }
  }

  // 이미 만들어진 파티클들을 한 프레임씩 움직이고, 다 수명이 끝난 건 지웁니다.
  function updateParticles() {
    effectCtx.clearRect(0, 0, effectCanvas.width, effectCanvas.height);

    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      p.x += p.vx; // 가로로 이동합니다.
      p.y += p.vy; // 세로로 이동합니다.
      p.vy += p.gravity; // 시간이 갈수록 아래로 떨어지게 만듭니다.
      p.vx *= 0.98; // 옆으로 가는 속도는 조금씩 줄입니다.
      p.life -= 1; // 수명도 한 프레임씩 줄어듭니다.

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      const alpha = p.life / p.maxLife;
      effectCtx.save();
      effectCtx.globalAlpha = alpha;
      effectCtx.fillStyle = p.color;
      effectCtx.beginPath();
      effectCtx.ellipse(p.x, p.y, p.size, p.size * 0.58, 0, 0, Math.PI * 2);
      effectCtx.fill();
      effectCtx.restore();
    }
  }

  return {
    spawnBurst,
    updateParticles
  };
}

// 같은 CSS 애니메이션을 다시 시작하려면 클래스를 한 번 빼고 강제로 재계산한 뒤 다시 붙여야 합니다.
export function restartClassAnimation(element, className) {
  if (!element) return;
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}
