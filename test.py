from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Launch the browser (headless=False lets you see the browser open)
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # Navigate to a URL
    page.goto("https://playwright.dev")
    print(f"Page Title: {page.title()}")

    # Take a screenshot
    page.screenshot(path="playwright_home.png")
    page.wait_for_load_state("networkidle")
    elements = page.locator("button").all()

    for element in elements:
        print(element.evaluate("el => el.tagName"))
    # Clean up
    browser.close()
