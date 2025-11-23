import re
import time
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc


def setup_driver_with_performance_log():
    """
    Setup undetected_chromedriver with performance logging
    (This captures network traffic like DevTools Network tab)
    """
    options = uc.ChromeOptions()

    # Enable performance logging (captures network like HAR)
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = uc.Chrome(options=options)

    print("✓ Browser started with Performance Logging (= DevTools Network Tab)")

    return driver


def handle_cookie_consent(driver, debug=False):
    """Handle cookie consent popup"""
    try:
        wait = WebDriverWait(driver, 5)

        button_selectors = [
            "//button[contains(translate(., 'AGREE', 'agree'), 'agree')]",
            "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
        ]

        for selector in button_selectors:
            try:
                button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                button.click()
                if debug:
                    print("✓ Clicked ACCEPT button")
                time.sleep(2)
                return True
            except:
                continue

        if debug:
            print("ℹ No consent button found")
        return False
    except Exception as e:
        if debug:
            print(f"⚠ Cookie consent error: {e}")
        return False


def convert_performance_log_to_har(performance_log, debug=False):
    """
    Convert Chrome Performance Log to HAR format
    """
    print("   Converting Performance Log to HAR format...")

    har = {
        "log": {
            "version": "1.2",
            "creator": {"name": "Python HAR Exporter", "version": "1.0"},
            "entries": []
        }
    }

    # Track requests by ID
    requests = {}

    for entry in performance_log:
        try:
            message = json.loads(entry['message'])
            method = message.get('message', {}).get('method', '')
            params = message.get('message', {}).get('params', {})

            # Network request sent
            if method == 'Network.requestWillBeSent':
                request_id = params.get('requestId')
                request = params.get('request', {})

                requests[request_id] = {
                    'request': request,
                    'response': None,
                    'timestamp': params.get('timestamp', 0)
                }

            # Response received
            elif method == 'Network.responseReceived':
                request_id = params.get('requestId')
                if request_id in requests:
                    requests[request_id]['response'] = params.get('response', {})

        except:
            continue

    # Convert to HAR entries
    for request_id, data in requests.items():
        if not data.get('request'):
            continue

        request = data['request']
        response = data.get('response') or {}  # Handle None response

        har_entry = {
            "request": {
                "method": request.get('method', 'GET'),
                "url": request.get('url', ''),
                "headers": [
                    {"name": k, "value": v}
                    for k, v in request.get('headers', {}).items()
                ]
            },
            "response": {
                "status": response.get('status', 0) if response else 0,
                "headers": [
                    {"name": k, "value": v}
                    for k, v in response.get('headers', {}).items()
                ] if response else []
            }
        }

        har['log']['entries'].append(har_entry)

    print(f"   ✓ Converted {len(har['log']['entries'])} network requests")

    return har


def analyze_har_for_coordinates(har_data, ship_id, debug=False):
    """
    Analyze HAR data for centerx/centery coordinates
    """
    print(f"\n{'='*70}")
    print(f"Analyzing HAR for centerx/centery coordinates")
    print(f"{'='*70}\n")

    entries = har_data['log']['entries']
    print(f"Total HAR entries: {len(entries)}")

    found_coords = []

    for i, entry in enumerate(entries):
        url = entry['request']['url']

        # Check URL for centerx/centery
        if 'centerx:' in url and 'centery:' in url:
            centerx_match = re.search(r'centerx:([-\d.]+)', url)
            centery_match = re.search(r'centery:([-\d.]+)', url)

            if centerx_match and centery_match:
                lon = float(centerx_match.group(1))
                lat = float(centery_match.group(1))

                found_coords.append({
                    'index': i,
                    'lat': lat,
                    'lon': lon,
                    'url': url,
                    'source': 'URL'
                })

                if debug:
                    print(f"[{i}] Found in URL: lat={lat}, lon={lon}")
                    print(f"    {url[:100]}...")

        # Check REFERER header
        for header in entry['request']['headers']:
            if header['name'].lower() == 'referer':
                referer = header['value']

                if 'centerx:' in referer and 'centery:' in referer:
                    centerx_match = re.search(r'centerx:([-\d.]+)', referer)
                    centery_match = re.search(r'centery:([-\d.]+)', referer)

                    if centerx_match and centery_match:
                        lon = float(centerx_match.group(1))
                        lat = float(centery_match.group(1))

                        found_coords.append({
                            'index': i,
                            'lat': lat,
                            'lon': lon,
                            'referer': referer,
                            'url': url,
                            'source': 'REFERER'
                        })

                        if debug:
                            print(f"[{i}] Found in REFERER: lat={lat}, lon={lon}")
                            print(f"    Referer: {referer[:100]}...")

    print(f"\n{'='*70}")
    print(f"RESULTS: Found {len(found_coords)} coordinate entries")
    print(f"{'='*70}\n")

    # Show unique coordinates
    unique_coords = {}
    for coord in found_coords:
        key = (coord['lat'], coord['lon'])
        if key not in unique_coords:
            unique_coords[key] = coord

    if unique_coords:
        print(f"Unique coordinates found:\n")
        for i, (key, coord) in enumerate(unique_coords.items()):
            print(f"[{i+1}] Lat: {coord['lat']}, Lon: {coord['lon']}")
            print(f"    Source: {coord['source']}")
            if 'referer' in coord:
                print(f"    Referer: {coord['referer'][:80]}...")
            else:
                print(f"    URL: {coord['url'][:80]}...")
            print()

        # Return the LAST (most specific/zoomed) coordinate
        # The last one is usually the ship position, earlier ones are map centers
        coords_list = list(unique_coords.values())

        if len(coords_list) > 1:
            print(f"ℹ Multiple coordinates found - returning LAST one (most specific)")
            print(f"  → Lat: {coords_list[-1]['lat']}, Lon: {coords_list[-1]['lon']}\n")
            return coords_list[-1]
        else:
            return coords_list[0]
    else:
        print("✗ No centerx/centery coordinates found in HAR")
        return None


def extract_coordinates_from_url_fallback(ship_id, debug=False):
    """
    Exact replication of manual HAR export process:

    0. Clear browser cache and storage (important!)
    1. Open DevTools Network Tab with "Preserve log" (Performance Logging)
    2. Load page with Ship-ID
    3. Click Cookie Accept
    4. RELOAD page (important!)
    5. Export HAR (get Performance Log)
    6. Analyze HAR for centerx/centery
    """
    driver = setup_driver_with_performance_log()

    try:
        map_url = f"https://www.marinetraffic.com/en/ais/home/shipid:{ship_id}/zoom:10"

        print(f"\n{'='*70}")
        print(f"HAR Export with Reload - Exact Manual Process")
        print(f"{'='*70}")
        print(f"Ship ID: {ship_id}\n")

        # Step 0: Clear browser cache and storage (CRITICAL!)
        print(f"Step 0: Clearing browser cache and storage...")
        driver.execute_cdp_cmd('Network.enable', {})
        driver.execute_cdp_cmd('Network.clearBrowserCache', {})
        driver.execute_cdp_cmd('Network.clearBrowserCookies', {})

        # Navigate to domain first to clear storage
        driver.get("https://www.marinetraffic.com")
        time.sleep(1)

        # Clear all storage
        driver.execute_script("""
            // Clear localStorage
            localStorage.clear();

            // Clear sessionStorage
            sessionStorage.clear();

            // Clear IndexedDB
            indexedDB.databases().then(dbs => {
                dbs.forEach(db => {
                    indexedDB.deleteDatabase(db.name);
                });
            });
        """)

        print(f"✓ Browser cache and storage cleared\n")

        # Step 1 & 2: Load page (Network Tab with Preserve Log active)
        print(f"Step 1-2: Loading page (Network capture active)...")
        print(f"URL: {map_url}")

        driver.get(map_url)
        time.sleep(3)
        print(f"✓ Page loaded\n")

        # Step 3: Click Cookie Accept
        print(f"Step 3: Clicking Cookie Accept...")
        handle_cookie_consent(driver, debug=True)
        time.sleep(2)
        print(f"✓ Cookie accepted\n")

        # Step 4: RELOAD page (CRITICAL - this generates the traffic we need!)
        print(f"Step 4: RELOADING page (this captures fresh network traffic)...")
        driver.refresh()
        time.sleep(5)  # Wait for all requests to complete
        print(f"✓ Page reloaded\n")

        # Step 5: Export HAR (get Performance Log)
        print(f"Step 5: Exporting HAR (retrieving Performance Log)...")
        performance_log = driver.get_log('performance')
        print(f"✓ Retrieved {len(performance_log)} performance log entries\n")

        # Convert to HAR format
        print(f"Converting to HAR format...")
        har_data = convert_performance_log_to_har(performance_log, debug=debug)

        # Save HAR file
        har_filename = f'marinetraffic_{ship_id}_har.json'
        with open(har_filename, 'w', encoding='utf-8') as f:
            json.dump(har_data, f, indent=2)
        print(f"✓ HAR file saved: {har_filename}\n")

        # Step 6: Analyze HAR
        print(f"Step 6: Analyzing HAR for centerx/centery...")
        coords = analyze_har_for_coordinates(har_data, ship_id, debug=debug)

        if coords:
            return coords
        else:
            print(f"\nℹ HAR file saved for manual inspection: {har_filename}")
            return None

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ EXCEPTION:")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        print(f"\nClosing browser...")
        driver.quit()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  HAR Export with Reload - Exact Manual DevTools Process          ║
    ║                                                                   ║
    ║  Steps (exactly as manual process):                               ║
    ║  1. DevTools Network Tab with "Preserve log" ON                  ║
    ║  2. Load page with Ship-ID                                        ║
    ║  3. Click Cookie ACCEPT                                           ║
    ║  4. RELOAD page (F5)                                              ║
    ║  5. Export HAR                                                    ║
    ║  6. Analyze for centerx/centery                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    SHIP_ID = 268855

    coords = extract_coordinates_from_url_fallback(SHIP_ID, debug=True)

    print(f"\n{'='*70}")
    if coords:
        print(f"✓✓✓ SUCCESS - Coordinates Found! ✓✓✓")
        print(f"Latitude:  {coords['lat']}")
        print(f"Longitude: {coords['lon']}")
        print(f"Source: {coords['source']}")
    else:
        print(f"✗ No coordinates found in HAR")
        print(f"Check marinetraffic_{SHIP_ID}_har.json for manual inspection")
    print(f"{'='*70}")