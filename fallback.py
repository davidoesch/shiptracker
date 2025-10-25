from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time
import requests

def setup_driver(headless=True):
    """Set up Chrome driver with proper options for GitHub Actions."""
    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument('--headless=new')

    # Essential for GitHub Actions
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # Suppress logging
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # Accept cookies/terms automatically
    prefs = {
        "profile.default_content_setting_values.cookies": 1,
        "profile.cookie_controls_mode": 0
    }
    options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("\nMake sure you have:")
        print("1. Chrome browser installed")
        print("2. ChromeDriver installed (pip install webdriver-manager)")
        print("3. Or use: from selenium.webdriver.chrome.service import Service")
        raise


def handle_cookie_consent(driver, debug=False):
    """
    Handle cookie consent popup if it appears.
    Tries multiple common selectors for AGREE/ACCEPT buttons.
    """
    try:
        wait = WebDriverWait(driver, 5)

        # Common button texts and selectors for cookie consent
        button_selectors = [
            # By button text
            "//button[contains(translate(., 'AGREE', 'agree'), 'agree')]",
            "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
            "//button[contains(translate(., 'CONSENT', 'consent'), 'consent')]",
            "//a[contains(translate(., 'AGREE', 'agree'), 'agree')]",
            "//a[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
            # By common class names
            "//button[contains(@class, 'accept')]",
            "//button[contains(@class, 'agree')]",
            "//button[contains(@class, 'consent')]",
            # By ID
            "//button[@id='cookie-accept']",
            "//button[@id='accept-cookies']",
            # Generic cookie banner buttons
            "//*[@class='cc-btn' or @class='cookie-btn']",
        ]

        for selector in button_selectors:
            try:
                button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                if debug:
                    print(f"Found consent button with selector: {selector}")

                current_url = driver.current_url
                button.click()
                if debug:
                    print("Clicked AGREE button")

                time.sleep(2)

                new_url = driver.current_url
                if new_url != current_url:
                    if debug:
                        print(f"Page redirected from {current_url} to {new_url}")
                    time.sleep(2)

                return True
            except:
                continue

        if debug:
            print("No consent button found (might not be needed)")
        return False

    except Exception as e:
        if debug:
            print(f"Cookie consent handling: {e}")
        return False


def extract_from_javascript(url, debug=False, max_retries=3):
    """
    Extract coordinates by analyzing the page's JavaScript.
    Looks for map initialization code with retry logic.
    """
    driver = setup_driver(headless=not debug)

    try:
        if debug:
            print(f"Loading URL: {url}")

        driver.get(url)
        handle_cookie_consent(driver, debug)

        if debug:
            print(f"Current URL after consent: {driver.current_url}")

        # Wait for map container to be present with longer timeout
        try:
            if debug:
                print("Waiting for map container...")

            map_container = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "map"))
            )
            if debug:
                print("Map container found!")
        except TimeoutException:
            if debug:
                print("Map container not found with ID='map', trying alternative selectors...")

            alternative_selectors = [
                (By.CLASS_NAME, "leaflet-container"),
                (By.CSS_SELECTOR, "[class*='map']"),
                (By.CSS_SELECTOR, "div[id*='map']"),
            ]

            map_found = False
            for by, selector in alternative_selectors:
                try:
                    map_container = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    if debug:
                        print(f"Map found with selector: {by}={selector}")
                    map_found = True
                    break
                except:
                    continue

            if not map_found:
                if debug:
                    print("No map container found at all!")
                return None

        time.sleep(3)

        # Retry logic for accessing the map object
        for attempt in range(max_retries):
            try:
                if debug:
                    print(f"Attempt {attempt + 1}/{max_retries} to access map object...")

                center = driver.execute_script("""
                    try {
                        if (typeof L === 'undefined') {
                            return {error: 'Leaflet not loaded'};
                        }

                        if (typeof map !== 'undefined' && map && map.getCenter) {
                            var center = map.getCenter();
                            return {lat: center.lat, lng: center.lng, method: 'global_map'};
                        }

                        if (L._maps) {
                            for (var id in L._maps) {
                                var mapInstance = L._maps[id];
                                if (mapInstance && mapInstance.getCenter) {
                                    var center = mapInstance.getCenter();
                                    return {lat: center.lat, lng: center.lng, method: 'L._maps'};
                                }
                            }
                        }

                        for (var key in window) {
                            try {
                                if (window[key] &&
                                    typeof window[key] === 'object' &&
                                    window[key].getCenter &&
                                    typeof window[key].getCenter === 'function') {
                                    var center = window[key].getCenter();
                                    if (center && center.lat !== undefined) {
                                        return {lat: center.lat, lng: center.lng, method: 'window_search'};
                                    }
                                }
                            } catch(e) {
                                continue;
                            }
                        }

                        return {error: 'No map instance found'};
                    } catch(e) {
                        return {error: e.toString()};
                    }
                """)

                if debug:
                    print(f"JavaScript result: {center}")

                if center and 'lat' in center and 'lng' in center:
                    if debug:
                        print(f"✓ Found coordinates via method: {center.get('method', 'unknown')}")
                    return {
                        'lat': center['lat'],
                        'lon': center['lng']
                    }
                elif center and 'error' in center:
                    if debug:
                        print(f"Error from JavaScript: {center['error']}")

                    if 'not loaded' in center['error'].lower():
                        if debug:
                            print("Waiting for Leaflet to load...")
                        time.sleep(3)
                        continue

                if attempt < max_retries - 1:
                    if debug:
                        print(f"Waiting before retry...")
                    time.sleep(3)

            except Exception as e:
                if debug:
                    print(f"Attempt {attempt + 1} exception: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)

        if debug:
            print("JavaScript methods failed, trying page source parsing...")

        page_source = driver.page_source

        patterns = [
            (r'setView\s*\(\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', 'setView'),
            (r'"lat"\s*:\s*([-\d.]+)\s*,\s*"lng"\s*:\s*([-\d.]+)', 'lat/lng JSON'),
            (r'"latitude"\s*:\s*([-\d.]+)\s*,\s*"longitude"\s*:\s*([-\d.]+)', 'latitude/longitude JSON'),
            (r'center:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', 'center array'),
        ]

        for pattern, name in patterns:
            match = re.search(pattern, page_source)
            if match:
                if debug:
                    print(f"Found coordinates via page source pattern: {name}")
                return {
                    'lat': float(match.group(1)),
                    'lon': float(match.group(2))
                }

        return None

    finally:
        driver.quit()


def extract_ship_details(details_url, debug=False):
    """
    Extract ship details from the details page using BeautifulSoup.

    Returns a dictionary with:
    - nav_status: 0 if "Underway" in status, 1 otherwise
    - timestamp_utc: formatted timestamp
    - sog: speed over ground (float)
    - cog: course over ground (int)
    """
    if debug:
        print(f"\nExtracting ship details from: {details_url}")

    driver = setup_driver(headless=not debug)

    try:
        driver.get(details_url)
        handle_cookie_consent(driver, debug)

        # Wait for the table to load
        time.sleep(5)

        # Get page source
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Initialize results
        result = {
            'nav_status': None,
            'timestamp_utc': None,
            'sog': None,
            'cog': None
        }

        # Find all table rows
        rows = soup.find_all('tr', class_='MuiTableRow-root')

        for row in rows:
            th = row.find('th', class_='MuiTableCell-root')
            td = row.find('td', class_='MuiTableCell-root')

            if not th or not td:
                continue

            header = th.get_text(strip=True)
            value = td.get_text(strip=True)

            if debug:
                print(f"Found: {header} = {value}")

            # Extract Navigational status
            if 'Navigational status' in header:
                result['nav_status'] = 0 if 'Underway' in value else 1
                if debug:
                    print(f"  → nav_status = {result['nav_status']}")

            # Extract Vessel's local time
            elif "Vessel's local time" in header or 'local time' in header.lower():
                # Parse time like "2025-10-25 14:40 (UTC+0)"
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', value)
                if time_match:
                    time_str = time_match.group(1)
                    try:
                        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
                        # Format as "2025-09-28 18:51:24.44833531 +0000 UTC"
                        result['timestamp_utc'] = dt.strftime('%Y-%m-%d %H:%M:%S.00000000 +0000 UTC')
                        if debug:
                            print(f"  → timestamp_utc = {result['timestamp_utc']}")
                    except Exception as e:
                        if debug:
                            print(f"  → Error parsing time: {e}")

            # Extract Speed
            elif 'Speed' in header:
                # Extract number from "6.5 kn"
                speed_match = re.search(r'([\d.]+)', value)
                if speed_match:
                    result['sog'] = float(speed_match.group(1))
                    if debug:
                        print(f"  → sog = {result['sog']}")

            # Extract Course
            elif 'Course' in header:
                # Extract number from "238 °"
                course_match = re.search(r'(\d+)', value)
                if course_match:
                    result['cog'] = int(course_match.group(1))
                    if debug:
                        print(f"  → cog = {result['cog']}")

        return result

    finally:
        driver.quit()


def get_ship_data(ship_id, debug=False):
    """
    Complete function to extract both coordinates and ship details.

    Args:
        ship_id: MarineTraffic ship ID
        debug: Enable debug output

    Returns:
        Dictionary with all extracted data
    """
    # Step 1: Extract coordinates from map
    map_url = f"https://www.marinetraffic.com/en/ais/home/shipid:{ship_id}/zoom:10"
    print(f"Step 1: Extracting coordinates from map...")
    coords = extract_from_javascript(map_url, debug=debug, max_retries=5)

    if not coords:
        print("✗ Failed to extract coordinates")
        return None

    print(f"✓ Coordinates extracted: {coords['lat']}, {coords['lon']}")

    # Step 2: Extract ship details
    details_url = f"https://www.marinetraffic.com/en/ais/details/ships/shipid:{ship_id}/"
    print(f"\nStep 2: Extracting ship details...")
    details = extract_ship_details(details_url, debug=debug)

    # Combine results
    result = {
        'lat': round(coords['lat'], 5),
        'lon': round(coords['lon'], 5),
        'nav_status': details['nav_status'],
        'timestamp_utc': details['timestamp_utc'],
        'sog': details['sog'],
        'cog': details['cog']
    }

    return result


# Example usage
if __name__ == "__main__":
    import sys

    # Check for debug flag
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv

    # Default ship ID
    ship_id = "268855"

    # Allow custom ship ID from command line
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        ship_id = sys.argv[1]

    print("=" * 60)
    print("MarineTraffic Ship Data Extractor")
    print("=" * 60)
    print(f"Ship ID: {ship_id}")
    print(f"Debug mode: {debug_mode}\n")

    # Extract all data
    data = get_ship_data(ship_id, debug=debug_mode)

    if data:
        print("\n" + "=" * 60)
        print("✓ SUCCESS - Complete Ship Data:")
        print("=" * 60)
        print(f"Latitude:           {data['lat']}")
        print(f"Longitude:          {data['lon']}")
        print(f"Navigational Status: {data['nav_status']} ({'Underway' if data['nav_status'] == 0 else 'Not Underway'})")
        print(f"Timestamp (UTC):    {data['timestamp_utc']}")
        print(f"Speed (SOG):        {data['sog']} kn")
        print(f"Course (COG):       {data['cog']}°")
        print("=" * 60)
    else:
        print("\n✗ FAILED - Could not extract ship data")
        print("=" * 60)