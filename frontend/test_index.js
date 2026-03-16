// [test_index.js] 메인 화면이 열리고 핵심 버튼/악기 요소가 존재하는지 확인하는 스모크 테스트입니다.

// puppeteer는 브라우저를 자동으로 열고 클릭/검사를 해주는 도구입니다.
const puppeteer = require("puppeteer");

// 비동기 작업을 바로 실행하는 함수입니다.
(async () => {
  try {
    // 화면 없이 백그라운드 브라우저를 엽니다.
    const browser = await puppeteer.launch({ headless: "new" });
    // 새 탭을 하나 엽니다.
    const page = await browser.newPage();

    // 페이지 안 콘솔 메시지를 터미널에도 보여줍니다.
    page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));
    // 페이지 자체 오류도 터미널에 출력합니다.
    page.on("pageerror", (error) => console.error("PAGE ERROR:", error.message));

    // 카메라 권한 팝업이 테스트를 막지 않도록 미리 허용합니다.
    const context = browser.defaultBrowserContext();
    await context.overridePermissions("http://localhost:3002", ["camera"]);

    // 메인 화면으로 이동합니다.
    console.log("Navigating to index.html");
    await page.goto("http://localhost:3002/index.html", { waitUntil: "networkidle2" });

    // 초기화가 끝날 시간을 조금 줍니다.
    console.log("Waiting 2 seconds for init...");
    await new Promise((r) => setTimeout(r, 2000));

    // 시작 버튼이 있는지 확인합니다.
    const startButton = await page.$("#landingStartButton");
    if (!startButton) {
      throw new Error("Missing #landingStartButton");
    }

    // 소리 버튼도 있는지 확인합니다.
    const soundButton = await page.$("#soundUnlockButton");
    if (!soundButton) {
      throw new Error("Missing #soundUnlockButton");
    }

    // 악기 버튼들이 전부 존재하는지 검사합니다.
    const instrumentIds = [
      "#instrumentDrum",
      "#instrumentXylophone",
      "#instrumentTambourine",
      "#instrumentA"
    ];

    for (const selector of instrumentIds) {
      const el = await page.$(selector);
      if (!el) {
        throw new Error(`Missing instrument element: ${selector}`);
      }
    }

    // 시작 버튼을 실제로 눌러봅니다.
    console.log("Clicking start button...");
    await startButton.click();
    await new Promise((r) => setTimeout(r, 500));

    // 시작 뒤에는 덮개 화면이 숨겨져야 정상입니다.
    const overlayHidden = await page.$eval("#landingOverlay", (el) => el.classList.contains("is-hidden"));
    if (!overlayHidden) {
      throw new Error("Landing overlay did not hide after start click");
    }

    // 여기까지 오면 가장 기본적인 메인 화면 진입은 성공입니다.
    console.log("Index smoke test passed");
    await browser.close();
  } catch (e) {
    // 어디서 실패했는지 에러를 보여줍니다.
    console.error("Puppeteer script failed:", e);
  }
})();
