// [particle_system.js] 악기 반응 시 튀는 파티클과 재시작형 애니메이션 효과를 맡는 모듈입니다.

// createParticleSystem(...) 은 파티클 목록을 내부에 숨긴 채,
// "추가하기(spawnBurst)" 와 "업데이트하기(updateParticles)" 만 밖으로 꺼내주는 공장 함수입니다.
export function createParticleSystem(effectCtx, effectCanvas) {
  let particles = [];
  const MAX_BUBBLES = 10; // 동시에 존재할 수 있는 최대 비눗방울 개수
  const MAX_PARTICLES = 90; // 전체 파티클 개수 제한

  function getCenterPointFromTarget(target) {
    if (!target || typeof target.getBoundingClientRect !== "function") return null;
    const rect = target.getBoundingClientRect();
    return {
      x: rect.left + rect.width * 0.5,
      y: rect.top + rect.height * 0.5
    };
  }

  function drawSparkParticle(particle, alpha) {
    effectCtx.save();
    effectCtx.globalAlpha = alpha;
    effectCtx.translate(particle.x, particle.y);
    effectCtx.rotate((particle.maxLife - particle.life) * 0.08);

    effectCtx.fillStyle = particle.color;
    effectCtx.beginPath();
    effectCtx.arc(0, 0, particle.size, 0, Math.PI * 2);
    effectCtx.fill();

    effectCtx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    effectCtx.lineWidth = Math.max(1, particle.size * 0.2);
    effectCtx.beginPath();
    effectCtx.moveTo(-particle.size * 1.4, 0);
    effectCtx.lineTo(particle.size * 1.4, 0);
    effectCtx.moveTo(0, -particle.size * 1.4);
    effectCtx.lineTo(0, particle.size * 1.4);
    effectCtx.stroke();

    effectCtx.restore();
  }

  // 특정 악기나 제스처 타입에 맞는 색과 양으로 파티클을 생성합니다.
  function spawnBurst(type, element) {
    const center = getCenterPointFromTarget(element);
    if (!center) return;
    const cx = center.x;
    const cy = center.y;
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
      type === "animal" ? 20
        : type === "kheart" ? 24
          : type === "fist" ? 18
            : 14;

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

  function spawnPointerTrail(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const baseColors = ["#f4e4b7", "#c2d6c0", "#d8c7ef", "#f0c8c8"];
    const count = 4;

    for (let i = 0; i < count; i += 1) {
      particles.push({
        x: x + (Math.random() - 0.5) * 8,
        y: y + (Math.random() - 0.5) * 8,
        vx: (Math.random() - 0.5) * 0.9,
        vy: -0.55 - Math.random() * 0.9,
        gravity: 0.01,
        life: 20 + Math.random() * 12,
        maxLife: 32,
        size: 1.8 + Math.random() * 2.2,
        color: baseColors[Math.floor(Math.random() * baseColors.length)]
      });
    }
  }

  function spawnPointerBurst(x, y) {
    spawnBurst("xylophone", {
      getBoundingClientRect: () => ({
        left: x - 10,
        top: y - 10,
        width: 20,
        height: 20
      })
    });
  }

  // 비눗방울은 화면 아래쪽에서 생성되어 위로 서서히 올라갑니다.
  function spawnBubble(canvasWidth, canvasHeight) {
    const currentBubbles = particles.filter(p => p.type === "bubble").length;
    if (currentBubbles >= MAX_BUBBLES) return;

    particles.push({
      type: "bubble",
      x: Math.random() * canvasWidth,
      y: canvasHeight + 25,
      vx: (Math.random() - 0.5) * 1.2,
      vy: -1.0 - Math.random() * 1.5,
      gravity: -0.012,
      life: 200 + Math.random() * 100,
      maxLife: 300,
      size: 12 + Math.random() * 20,
      color: "rgba(255, 255, 255, 0.35)",
      borderColor: "rgba(173, 216, 230, 0.7)"
    });
  }

  // 이미 만들어진 파티클들을 한 프레임씩 움직이고, 다 수명이 끝난 건 지웁니다.
  function updateParticles() {
    effectCtx.clearRect(0, 0, effectCanvas.width, effectCanvas.height);

    // 개수가 너무 많으면 오래된 것부터 정리합니다.
    if (particles.length > MAX_PARTICLES) {
      particles.splice(0, particles.length - MAX_PARTICLES);
    }

    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += p.gravity;
      p.vx *= 0.992;
      p.life -= 1;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      const alpha = p.life / p.maxLife;
      
      if (p.type === "bubble") {
        effectCtx.save();
        effectCtx.globalAlpha = alpha;
        // 비눗방울 그리기
        effectCtx.beginPath();
        effectCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        effectCtx.fillStyle = p.color;
        effectCtx.fill();
        effectCtx.strokeStyle = p.borderColor;
        effectCtx.lineWidth = 1.5;
        effectCtx.stroke();

        // 하이라이트
        effectCtx.beginPath();
        effectCtx.arc(p.x - p.size * 0.3, p.y - p.size * 0.3, p.size * 0.15, 0, Math.PI * 2);
        effectCtx.fillStyle = "rgba(255, 255, 255, 0.5)";
        effectCtx.fill();
        effectCtx.restore();
      } else {
        drawSparkParticle(p, alpha);
      }
    }
    // 루프 끝난 후 Alpha 복구
    effectCtx.globalAlpha = 1.0;
  }

  // 손가락 끝(point)들이 비눗방울과 닿았는지 확인하고 터뜨립니다.
  function checkBubbleCollision(points) {
    if (!Array.isArray(points)) points = [points];
    let poppedTotal = false;

    for (let i = particles.length - 1; i >= 0; i -= 1) {
      const p = particles[i];
      if (p.type !== "bubble") continue;

      for (const pt of points) {
        const dx = p.x - pt.x;
        const dy = p.y - pt.y;
        const distSq = dx * dx + dy * dy;
        const threshold = (p.size + 18) ** 2; // 제곱근 계산 방지

        if (distSq < threshold) {
          spawnBurst("xylophone", { getBoundingClientRect: () => ({ left: p.x - 4, top: p.y - 4, width: 8, height: 8 }) });
          particles.splice(i, 1);
          poppedTotal = true;
          break; // 이 비눗방울은 이미 터졌으므로 다음 포인트 검사 불필요
        }
      }
    }
    return poppedTotal;
  }

  return {
    spawnBurst,
    spawnPointerTrail,
    spawnPointerBurst,
    spawnBubble,
    updateParticles,
    checkBubbleCollision
  };
}

// 같은 CSS 애니메이션을 다시 시작하려면 클래스를 한 번 빼고 강제로 재계산한 뒤 다시 붙여야 합니다.
export function restartClassAnimation(element, className) {
  if (!element) return;
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}
