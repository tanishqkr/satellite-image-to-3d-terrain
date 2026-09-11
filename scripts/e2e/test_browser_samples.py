import os
import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def run_browser_tests():
    out_dir = ROOT_DIR / "browser_tests"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {
        "landing_page": {},
        "sample_1_geotiff": {},
        "sample_2_png": {},
        "console_errors": []
    }

    with sync_playwright() as p:
        # Launch browser with headless=False so user can see it live on screen
        browser = p.chromium.launch(channel="msedge", headless=False)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        def on_console(msg):
            t = msg.type
            txt = msg.text
            print(f"[BROWSER {t.upper()}]: {txt}")
            if t in ["error"]:
                results["console_errors"].append(txt)
        
        page.on("console", on_console)
        page.on("pageerror", lambda err: results["console_errors"].append(str(err)))
        page.on("response", lambda resp: print(f"[NET {resp.status}]: {resp.url}") if resp.status >= 400 else None)

        print("--- Navigating to http://localhost:5173/ ---")
        page.goto("http://localhost:5173/", wait_until="networkidle", timeout=30000)
        time.sleep(2)
        
        # 1. Landing Page Inspection
        title = page.title()
        results["landing_page"]["title"] = title
        landing_path = os.path.join(out_dir, "01_landing_neo_brutalism.png")
        page.screenshot(path=landing_path, full_page=True)
        print(f"Landing page screenshot saved: {landing_path}")

        tif_btn = page.locator("button:has-text('LOAD TIF')")
        png_btn = page.locator("button:has-text('LOAD PNG')")
        print(f"TIF button count: {tif_btn.count()}, PNG button count: {png_btn.count()}")

        # 2. Test SAMPLE 1 (Real GeoTIFF)
        print("\n--- Testing SAMPLE 1: Real GeoTIFF ---")
        tif_btn.click()
        
        print("Waiting for inference to complete...")
        page.wait_for_selector("text=NEW IMAGE", timeout=60000)
        time.sleep(4)

        header_text = page.locator("header").inner_text()
        print(f"Sample 1 Header content:\n{header_text}")
        results["sample_1_geotiff"]["header"] = header_text

        s1_view_path = os.path.join(out_dir, "02_sample1_geotiff_studio.png")
        page.screenshot(path=s1_view_path)
        print(f"Sample 1 Studio screenshot saved: {s1_view_path}")

        # Interact with Flood Inundation Simulator
        print("Testing Flood Inundation Simulator...")
        flood_slider = page.locator("input[type='range']").first
        if flood_slider.count() > 0:
            flood_slider.evaluate("el => { el.value = 45; el.dispatchEvent(new Event('input', { bubbles: true })); }")
            time.sleep(1)
            try:
                flood_card = page.locator("text=FLOOD INUNDATION SIMULATOR").locator("xpath=ancestor::div[contains(@class, 'neo-card')]")
                print(f"Flood card content:\n{flood_card.inner_text()}")
                results["sample_1_geotiff"]["flood_section"] = flood_card.inner_text()
            except Exception as e:
                print(f"Flood section reading note: {e}")
            
            s1_flood_path = os.path.join(out_dir, "03_sample1_flood_active.png")
            page.screenshot(path=s1_flood_path)
            print(f"Sample 1 Flood screenshot saved: {s1_flood_path}")

        # Expand Cross-Section Profile
        print("Testing Cross-Section Profile...")
        cs_expand = page.locator("button:has-text('EXPAND')")
        if cs_expand.count() > 0:
            cs_expand.click()
            time.sleep(1)
            s1_cs_path = os.path.join(out_dir, "04_sample1_cross_section_expanded.png")
            page.screenshot(path=s1_cs_path)
            print(f"Sample 1 Cross Section screenshot saved: {s1_cs_path}")

        # 3. Return and Test SAMPLE 2 (Urban Optical PNG)
        print("\n--- Returning to dropzone for SAMPLE 2 ---")
        new_img_btn = page.locator("button:has-text('NEW IMAGE')")
        new_img_btn.click()
        time.sleep(1)
        
        print("Testing SAMPLE 2: Urban Optical PNG...")
        png_btn = page.locator("button:has-text('LOAD PNG')")
        png_btn.click()

        print("Waiting for PNG inference to complete...")
        page.wait_for_selector("text=NEW IMAGE", timeout=60000)
        time.sleep(5) # Let user observe the 3D relief on screen!

        header_text_2 = page.locator("header").inner_text()
        print(f"Sample 2 Header content:\n{header_text_2}")
        results["sample_2_png"]["header"] = header_text_2

        s2_view_path = os.path.join(out_dir, "05_sample2_png_studio.png")
        page.screenshot(path=s2_view_path)
        print(f"Sample 2 Studio screenshot saved: {s2_view_path}")

        # Expand Cross-Section on Sample 2
        cs_expand2 = page.locator("button:has-text('EXPAND')")
        if cs_expand2.count() > 0:
            cs_expand2.click()
            time.sleep(3)
            s2_cs_path = os.path.join(out_dir, "06_sample2_cross_section.png")
            page.screenshot(path=s2_cs_path)
            print(f"Sample 2 Cross Section screenshot saved: {s2_cs_path}")

        print("Keeping browser open for 6 seconds for user viewing...")
        time.sleep(6)

        summary_file = os.path.join(out_dir, "browser_test_results.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nAll browser tests finished successfully! Summary saved to: {summary_file}")

        browser.close()

if __name__ == '__main__':
    run_browser_tests()
