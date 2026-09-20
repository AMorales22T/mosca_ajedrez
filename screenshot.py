import asyncio
from playwright.async_api import async_playwright


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        for view in ("side", "dorsal", "front"):
            await page.evaluate("view => setCameraView(view)", view)
            await page.wait_for_timeout(600)
            await page.screenshot(path=f"screenshot_{view}.png", full_page=True)
        await browser.close()


asyncio.run(take_screenshots())
