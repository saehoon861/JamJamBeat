// [test_theme.js] 이 파일은 화면이 잘 열리고 버튼이 잘 눌리는지 자동으로 확인하는 테스트 파일입니다.
// 사람이 직접 브라우저를 열어보지 않아도, 컴퓨터가 대신 실행해 보면서 문제가 없는지 검사합니다.

// puppeteer는 크롬 브라우저를 자동으로 열고 조작하게 해주는 도구입니다.
const puppeteer = require('puppeteer');

// (async () => { ... })() 형태는 "비동기 작업을 바로 실행"할 때 자주 쓰는 문법입니다.
// 여기서는 브라우저 열기, 페이지 접속, 클릭 같은 순서를 차례대로 기다리며 실행하려고 사용합니다.
(async () => {
    // try는 "문제가 생길 수 있는 작업"을 감싸는 부분입니다.
    // 아래 작업 중 에러가 나면 catch로 넘어가서 원인을 보여줍니다.
    try {
        // headless: 'new' 는 브라우저를 화면 없이 백그라운드에서 실행하라는 뜻입니다.
        // 즉, 실제 창을 띄우지 않고도 테스트를 할 수 있습니다.
        const browser = await puppeteer.launch({ headless: 'new' });
        // 새 탭 하나를 엽니다.
        const page = await browser.newPage();

        // 웹페이지 안에서 console.log가 실행되면 그 내용을 터미널에도 보여줍니다.
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        // 웹페이지 자체에서 오류가 나면 그 오류 메시지를 터미널에 보여줍니다.
        page.on('pageerror', error => console.error('PAGE ERROR:', error.message));

        // 카메라 권한 요청 창이 뜨면 테스트가 멈출 수 있으므로,
        // localhost:3002 주소에는 카메라 권한을 미리 허용합니다.
        const context = browser.defaultBrowserContext();
        await context.overridePermissions('http://localhost:3002', ['camera']);

        // 어떤 페이지로 이동하는지 터미널에 알려줍니다.
        console.log("Navigating to theme.html");
        // theme.html 페이지를 엽니다.
        // networkidle2는 네트워크 요청이 거의 멈출 때까지 기다리라는 뜻이라,
        // 화면이 어느 정도 다 로드된 뒤 다음 단계로 넘어가게 도와줍니다.
        await page.goto('http://localhost:3002/theme.html', { waitUntil: 'networkidle2' });

        // 페이지가 열린 직후 바로 누르면 아직 준비가 덜 됐을 수 있으니 2초 기다립니다.
        console.log("Waiting 2 seconds for init...");
        // setTimeout을 Promise로 감싸서 "지정한 시간만큼 잠깐 멈추기"를 구현합니다.
        await new Promise(r => setTimeout(r, 2000));

        // 화면 안에서 .theme-card 라는 요소를 찾아 첫 번째 카드를 눌러보려는 단계입니다.
        console.log("Attempting to click first theme card...");
        // page.$$는 CSS 선택자로 해당 요소들을 전부 배열처럼 가져옵니다.
        const cards = await page.$$('.theme-card');
        // 카드가 하나라도 있으면 첫 번째 카드를 클릭합니다.
        if (cards.length > 0) {
            await cards[0].click();
            console.log("Clicked card!");
        } else {
            // 카드가 아예 없으면, 화면 구조가 예상과 다르다는 뜻이므로 로그를 남깁니다.
            console.log("No theme cards found!");
        }

        // 클릭 뒤에 화면 이동이나 애니메이션이 있을 수 있으니 1초 더 기다립니다.
        console.log("Waiting 1 second after click...");
        await new Promise(r => setTimeout(r, 1000));
        // 클릭 후 현재 주소(URL)가 어디로 바뀌었는지 확인해서 결과를 기록합니다.
        console.log("Current URL:", page.url());

        // 테스트가 끝났으므로 브라우저를 닫아 자원을 정리합니다.
        await browser.close();
    } catch (e) {
        // 위 과정 어디에서든 에러가 나면 여기서 한 번에 잡아서 출력합니다.
        console.error("Puppeteer script failed:", e);
    }
    // 함수 정의 뒤의 ()는 "지금 바로 실행"하라는 뜻입니다.
})();
