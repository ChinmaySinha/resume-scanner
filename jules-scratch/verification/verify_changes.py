
from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Upload resume
        page.set_input_files('input[type="file"]', 'resume.txt')

        # Fill in job title and description
        page.fill('input[type="text"]', "Software Engineer")
        page.fill('textarea', "python, java, sql")

        # Click process button
        page.click('button:has-text("Process / Score")')

        # Wait for the "Extracted skills" subheader to be visible
        subheader = page.locator("h2", has_text="Extracted skills")
        expect(subheader).to_be_visible(timeout=180000)

        # Take screenshot of the final state
        page.screenshot(path="jules-scratch/verification/verification.png")

        browser.close()

if __name__ == "__main__":
    run()
