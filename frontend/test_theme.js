const puppeteer = require('puppeteer');

(async () => {
    try {
        const browser = await puppeteer.launch({ headless: 'new' });
        const page = await browser.newPage();

        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        page.on('pageerror', error => console.error('PAGE ERROR:', error.message));

        // Grant camera permissions to avoid prompts blocking
        const context = browser.defaultBrowserContext();
        await context.overridePermissions('http://localhost:3002', ['camera']);

        console.log("Navigating to theme.html");
        await page.goto('http://localhost:3002/theme.html', { waitUntil: 'networkidle2' });

        console.log("Waiting 2 seconds for init...");
        await new Promise(r => setTimeout(r, 2000));

        console.log("Attempting to click first theme card...");
        const cards = await page.$$('.theme-card');
        if (cards.length > 0) {
            await cards[0].click();
            console.log("Clicked card!");
        } else {
            console.log("No theme cards found!");
        }

        console.log("Waiting 1 second after click...");
        await new Promise(r => setTimeout(r, 1000));
        console.log("Current URL:", page.url());

        await browser.close();
    } catch (e) {
        console.error("Puppeteer script failed:", e);
    }
})();
