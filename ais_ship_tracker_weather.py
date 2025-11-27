import asyncio
import websockets
import json
import csv
import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import time
import config
from dateutil import parser as date_parser
import undetected_chromedriver as uc
import math
from geopy.distance import geodesic


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_timestamp_with_tz(timestamp_str):
    """Parse timestamp string preserving timezone information."""
    try:
        if ' UTC' in timestamp_str and '+' in timestamp_str:
            timestamp_str = timestamp_str.replace(' UTC', '')
            parts = timestamp_str.rsplit('+', 1)
            if len(parts) == 2:
                dt_part = parts[0].strip()
                tz_part = '+' + parts[1].strip()
                if '.' in dt_part:
                    dt_part = dt_part.split('.')[0]
                timestamp_str = f"{dt_part}{tz_part}"

        parsed_dt = date_parser.parse(timestamp_str)
        return parsed_dt.replace(microsecond=0)
    except:
        try:
            parsed_dt = datetime.fromisoformat(timestamp_str)
            return parsed_dt.replace(microsecond=0)
        except:
            try:
                dt = datetime.strptime(timestamp_str.split('+')[0].strip().split('.')[0], '%Y-%m-%d %H:%M:%S')
                return dt.replace(tzinfo=timezone.utc, microsecond=0)
            except:
                print(f"Warning: Could not parse timestamp '{timestamp_str}', using current UTC time")
                return datetime.now(timezone.utc).replace(microsecond=0)


def format_timestamp_with_tz(dt):
    """Format datetime object to ISO 8601 string preserving timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def get_api_key():
    """Get API key from environment variable or local secrets file."""
    api_key = os.getenv('AISSTREAM_API_KEY')
    if api_key:
        return api_key

    secrets_file = os.path.join('secrets', 'aisstream.json')
    if os.path.exists(secrets_file):
        try:
            with open(secrets_file, 'r') as f:
                secrets = json.load(f)
                return secrets.get('APIKey')
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading secrets file: {e}")
            return None

    print("API key not found in environment variables or secrets file")
    return None


def create_bounding_box_around_position(latitude, longitude, radius_km=200):
    """Create a bounding box around a given position."""
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))

    lat_min = max(-90, latitude - lat_delta)
    lat_max = min(90, latitude + lat_delta)
    lon_min = max(-180, longitude - lon_delta)
    lon_max = min(180, longitude + lon_delta)

    return [[[lat_min, lon_min], [lat_max, lon_max]]]


# ============================================================================
# WEATHER DATA FUNCTIONS
# ============================================================================

async def get_weather_data(latitude, longitude):
    """Retrieve atmospheric and marine weather data from Open-Meteo API."""
    try:
        weather_data = {}

        atmo_url = "https://api.open-meteo.com/v1/forecast"
        atmo_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["weather_code", "surface_pressure", "wind_direction_10m",
                       "wind_speed_10m", "cloud_cover", "temperature_2m", "precipitation"]
        }

        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["sea_surface_temperature", "wave_height"]
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(atmo_url, params=atmo_params) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get('current', {})
                    weather_code = current.get('weather_code', None)

                    weather_data.update({
                        'wetterzustand': get_weather_description(weather_code),
                        'luftdruck': current.get('surface_pressure', None),
                        'windrichtung': current.get('wind_direction_10m', None),
                        'windstaerke': current.get('wind_speed_10m', None),
                        'bewoelkung': current.get('cloud_cover', None),
                        'lufttemperatur': current.get('temperature_2m', None),
                        'niederschlag': current.get('precipitation', None)
                    })

            async with session.get(marine_url, params=marine_params) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get('current', {})
                    weather_data.update({
                        'wassertemperatur': current.get('sea_surface_temperature', None),
                        'wellenhoehe': current.get('wave_height', None)
                    })

        return weather_data if weather_data else None

    except Exception as e:
        print(f"Error retrieving weather data: {e}")
        return None


def get_weather_description(weather_code):
    """Convert WMO weather codes to German descriptions."""
    if weather_code is None:
        return None

    weather_codes = {
        0: "Sonnig", 1: "Überwiegend klar", 2: "Teilweise bewölkt", 3: "Bedeckt",
        45: "Nebel", 48: "Nebel mit Reifablagerung", 51: "Leichter Sprühregen",
        53: "Mäßiger Sprühregen", 55: "Starker Sprühregen", 61: "Leichter Regen",
        63: "Mäßiger Regen", 65: "Starker Regen", 71: "Leichter Schneefall",
        73: "Mäßiger Schneefall", 75: "Starker Schneefall", 80: "Leichte Regenschauer",
        81: "Mäßige Regenschauer", 82: "Starke Regenschauer", 95: "Gewitter"
    }

    return weather_codes.get(weather_code, f"Unbekannt ({weather_code})")


# ============================================================================
# WEB SCRAPING FUNCTIONS
# ============================================================================

def setup_driver(headless=True):
    """Setup Selenium WebDriver using undetected-chromedriver."""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument('--headless=new')

    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = uc.Chrome(options=options)
    return driver


def handle_cookie_consent(driver, debug=False):
    """Handle cookie consent popup if it appears."""
    try:
        wait = WebDriverWait(driver, 5)
        button_selectors = [
            "//button[contains(translate(., 'AGREE', 'agree'), 'agree')]",
            "//button[contains(translate(., 'ACCEPT', 'accept'), 'accept')]",
            "//button[contains(@class, 'accept')]",
            "//button[@id='cookie-accept']"
        ]

        for selector in button_selectors:
            try:
                button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                if debug:
                    print(f"Found consent button with selector: {selector}")
                button.click()
                time.sleep(2)
                return True
            except:
                continue

        if debug:
            print("No consent button found")
        return False

    except Exception as e:
        if debug:
            print(f"Cookie consent handling: {e}")
        return False


# ============================================================================
# OPTION 1: AIS STREAM
# ============================================================================

async def get_ship_data_from_aisstream(mmsi, max_duration_minutes, latest_entry_time=None):
    """
    Attempt to get ship position data from AIS Stream WebSocket.

    Returns:
        dict: Ship data with keys: timestamp_utc, mmsi, latitude, longitude,
              cog, sog, true_heading, nav_status, shipname
        None: If connection fails or no data received within time limit
    """
    api_key = get_api_key()
    if not api_key:
        print("Error: API key not found")
        return None

    try:
        async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
            subscribe_message = {
                "APIKey": api_key,
                "BoundingBoxes": config.BOUNDING_BOXES,
                "FiltersShipMMSI": [mmsi],
                "FilterMessageTypes": ["PositionReport"]
            }

            await websocket.send(json.dumps(subscribe_message))

            start_time = datetime.now(timezone.utc)
            max_duration = timedelta(minutes=max_duration_minutes)

            print(f"Starting AIS stream monitoring for {max_duration_minutes} minutes...")
            print(f"Monitoring MMSI: {mmsi}")

            while True:
                current_time = datetime.now(timezone.utc)

                if current_time - start_time > max_duration:
                    print(f"{max_duration_minutes} minutes elapsed. No data received.")
                    return None

                try:
                    message_json = await asyncio.wait_for(websocket.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue

                message = json.loads(message_json)

                if message["MessageType"] == "PositionReport":
                    ais_message = message['Message']['PositionReport']
                    meta_data = message['MetaData']

                    time_utc_from_report = meta_data.get('time_utc', current_time.isoformat())
                    parsed_time = parse_timestamp_with_tz(time_utc_from_report)

                    ship_data = {
                        'timestamp_utc': format_timestamp_with_tz(parsed_time),
                        'mmsi': ais_message.get('UserID', mmsi),
                        'latitude': round(ais_message.get('Latitude', 0), 5),
                        'longitude': round(ais_message.get('Longitude', 0), 5),
                        'cog': ais_message.get('Cog', 360),
                        'sog': ais_message.get('Sog', 0),
                        'true_heading': ais_message.get('TrueHeading', 511),
                        'nav_status': ais_message.get('NavigationalStatus', 15),
                        'shipname': meta_data.get('ShipName', 'Unknown')
                    }

                    print(f"✓ AIS Stream data received for MMSI {mmsi}")
                    return ship_data

    except websockets.exceptions.InvalidStatus as e:
        print(f"WebSocket connection failed: {e}")
        return None
    except Exception as e:
        print(f"WebSocket error: {e}")
        return None


# ============================================================================
# OPTION 2: SCRAPE SITE
# ============================================================================

def get_ship_data_from_scrapesite(ship_id, mmsi, debug=False, headless=False):
    """
    Scrape ship position data from MarineTraffic website.

    Args:
        ship_id: MarineTraffic ship ID
        mmsi: Ship MMSI number
        debug: Enable debug output
        headless: Run browser in headless mode (default: False for visibility)

    Returns:
        dict: Ship data with keys: timestamp_utc, mmsi, latitude, longitude,
              cog, sog, true_heading, nav_status, shipname
        None: If scraping fails
    """
    driver = setup_driver(headless=headless)

    try:
        # First, load main page to establish session and handle cookies
        main_url = f"https://www.marinetraffic.com/en/ais/home/shipid:{ship_id}/zoom:10"
        if debug:
            print(f"Loading main page: {main_url}")

        driver.get(main_url)
        handle_cookie_consent(driver, debug)
        time.sleep(3)

        # Now navigate to the API endpoint
        api_url = f"https://www.marinetraffic.com/map/getvesseljson/shipid:{ship_id}"
        if debug:
            print(f"Fetching API data: {api_url}")

        driver.get(api_url)
        time.sleep(2)

        # Get the JSON response
        raw_json_data = driver.page_source
        clean_json_text = re.sub(r'<[^>]*>', '', raw_json_data, flags=re.IGNORECASE).strip()

        if not clean_json_text.startswith('{'):
            if debug:
                print(f"Response is not JSON: {clean_json_text[:100]}")
            return None

        data = json.loads(clean_json_text)

        # Extract ship data from API response
        if 'LAT' not in data or 'LON' not in data:
            if debug:
                print("Missing LAT/LON in API response")
            return None

        # Parse timestamp
        timestamp_str = data.get('TIMESTAMP', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            dt = dt.replace(tzinfo=timezone.utc)
        except:
            dt = datetime.now(timezone.utc)

        # Determine navigational status (0 = underway, 1 = stopped/anchored)
        speed = float(data.get('SPEED', 0))
        nav_status = 0 if speed > 0.1 else 1

        ship_data = {
            'timestamp_utc': format_timestamp_with_tz(dt),
            'mmsi': data.get('MMSI', mmsi),
            'latitude': round(float(data['LAT']), 5),
            'longitude': round(float(data['LON']), 5),
            'cog': 360 if nav_status == 1 else int(float(data.get('COURSE', 360))),
            'sog': float(data.get('SPEED', 0)),
            'true_heading':511,
            'nav_status': nav_status,
            'shipname': data.get('SHIPNAME', 'Unknown')
        }

        if debug:
            print(f"✓ Scraped data successfully for MMSI {mmsi}")

        return ship_data

    except Exception as e:
        print(f"Scraping error: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        driver.quit()

# ============================================================================
# OPTION 3: HAR ANALYSIS
# ============================================================================

def setup_driver_with_performance_log():
    """Setup undetected_chromedriver with performance logging."""
    options = uc.ChromeOptions()
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    driver = uc.Chrome(options=options)
    print("✓ Browser started with Performance Logging")
    return driver


def convert_performance_log_to_har(performance_log, debug=False):
    """Convert Chrome Performance Log to HAR format."""
    print("   Converting Performance Log to HAR format...")

    har = {
        "log": {
            "version": "1.2",
            "creator": {"name": "Python HAR Exporter", "version": "1.0"},
            "entries": []
        }
    }

    requests = {}

    for entry in performance_log:
        try:
            message = json.loads(entry['message'])
            method = message.get('message', {}).get('method', '')
            params = message.get('message', {}).get('params', {})

            if method == 'Network.requestWillBeSent':
                request_id = params.get('requestId')
                request = params.get('request', {})
                requests[request_id] = {
                    'request': request,
                    'response': None,
                    'timestamp': params.get('timestamp', 0)
                }

            elif method == 'Network.responseReceived':
                request_id = params.get('requestId')
                if request_id in requests:
                    requests[request_id]['response'] = params.get('response', {})

        except:
            continue

    for request_id, data in requests.items():
        if not data.get('request'):
            continue

        request = data['request']
        response = data.get('response') or {}

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
    """Analyze HAR data for centerx/centery coordinates."""
    print(f"\n{'='*70}")
    print(f"Analyzing HAR for centerx/centery coordinates")
    print(f"{'='*70}\n")

    entries = har_data['log']['entries']
    print(f"Total HAR entries: {len(entries)}")

    found_coords = []

    for i, entry in enumerate(entries):
        url = entry['request']['url']

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

    print(f"\n{'='*70}")
    print(f"RESULTS: Found {len(found_coords)} coordinate entries")
    print(f"{'='*70}\n")

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
            print()

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


def get_ship_data_from_har_analysis(ship_id, mmsi, debug=False):
    """
    Extract ship coordinates using HAR analysis method.

    Args:
        ship_id: MarineTraffic ship ID
        mmsi: Ship MMSI number
        debug: Enable debug output

    Returns:
        dict: Ship data with keys: timestamp_utc, mmsi, latitude, longitude,
              cog, sog, true_heading, nav_status, shipname (most empty/default values)
        None: If extraction fails
    """
    driver = setup_driver_with_performance_log()

    try:
        map_url = f"https://www.marinetraffic.com/en/ais/home/shipid:{ship_id}/zoom:10"

        print(f"\n{'='*70}")
        print(f"HAR Export with Reload - Exact Manual Process")
        print(f"{'='*70}")
        print(f"Ship ID: {ship_id}\n")

        # Clear browser cache and storage
        print(f"Step 0: Clearing browser cache and storage...")
        driver.execute_cdp_cmd('Network.enable', {})
        driver.execute_cdp_cmd('Network.clearBrowserCache', {})
        driver.execute_cdp_cmd('Network.clearBrowserCookies', {})

        driver.get("https://www.marinetraffic.com")
        time.sleep(1)

        driver.execute_script("""
            localStorage.clear();
            sessionStorage.clear();
            indexedDB.databases().then(dbs => {
                dbs.forEach(db => {
                    indexedDB.deleteDatabase(db.name);
                });
            });
        """)

        print(f"✓ Browser cache and storage cleared\n")

        # Load page
        print(f"Step 1-2: Loading page (Network capture active)...")
        print(f"URL: {map_url}")
        driver.get(map_url)
        time.sleep(3)
        print(f"✓ Page loaded\n")

        # Handle cookies
        print(f"Step 3: Clicking Cookie Accept...")
        handle_cookie_consent(driver, debug=True)
        time.sleep(2)
        print(f"✓ Cookie accepted\n")

        # Reload page
        print(f"Step 4: RELOADING page (this captures fresh network traffic)...")
        driver.refresh()
        time.sleep(5)
        print(f"✓ Page reloaded\n")

        # Export HAR
        print(f"Step 5: Exporting HAR (retrieving Performance Log)...")
        performance_log = driver.get_log('performance')
        print(f"✓ Retrieved {len(performance_log)} performance log entries\n")

        # Convert to HAR
        print(f"Converting to HAR format...")
        har_data = convert_performance_log_to_har(performance_log, debug=debug)

        # Save HAR file
        har_filename = f'marinetraffic_{ship_id}_har.json'
        with open(har_filename, 'w', encoding='utf-8') as f:
            json.dump(har_data, f, indent=2)
        print(f"✓ HAR file saved: {har_filename}\n")

        # Analyze HAR
        print(f"Step 6: Analyzing HAR for centerx/centery...")
        coords = analyze_har_for_coordinates(har_data, ship_id, debug=debug)

        if coords:
            # Create ship_data dict with coordinates and empty/default values
            current_time = datetime.now(timezone.utc).replace(microsecond=0)

            ship_data = {
                'timestamp_utc': current_time.isoformat(),
                'mmsi': mmsi,
                'latitude': round(coords['lat'], 5),
                'longitude': round(coords['lon'], 5),
                'cog': 370,  # is no data value
                'sog': 5.5,  # Assuming some movement
                'true_heading': 511,  # Empty/default
                'nav_status': 8,  # IN Optione 3 we assume underway
                'shipname': ' '  # Empty/default
            }

            print(f"✓ HAR analysis successful for MMSI {mmsi}")
            try:
                if os.path.exists(har_filename):
                    os.remove(har_filename)

            except Exception as e:
                print(f"Could not delete HAR file {har_filename}: {e}")
            return ship_data
        else:
            print(f"\nℹ HAR file saved for manual inspection: {har_filename}")
            return None

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ HAR Analysis Exception:")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        print(f"\nClosing browser...")
        driver.quit()


# ============================================================================
# DATA PROCESSING AND STORAGE
# ============================================================================

async def process_and_save_ship_data(ship_data, latest_entry, latest_entry_time, csv_filename):
    """
    Process ship data, apply filters, get weather data, and save to CSV.

    Returns:
        tuple: (success, new_latest_entry_time, new_latest_entry, weather_data)
    """
    print(f"[{ship_data['timestamp_utc']}] MMSI: {ship_data['mmsi']}, "
          f"Lat: {ship_data['latitude']}, Lon: {ship_data['longitude']}, "
          f"SOG: {ship_data['sog']}, Nav Status: {ship_data['nav_status']}")

    # Get weather data
    print("Retrieving weather data...")
    weather_data = await get_weather_data(ship_data['latitude'], ship_data['longitude'])

    if weather_data:
        print(f"Wetter: {weather_data.get('wetterzustand', 'N/A')}, "
              f"Temp: {weather_data.get('lufttemperatur', 'N/A')}°C, "
              f"Wind: {weather_data.get('windstaerke', 'N/A')} km/h")

    # Check if we should add this entry
    should_add = True

    # Rule: Don't add if vessel not moving AND latest entry also not moving
    if ship_data['nav_status'] == 1 or ship_data['sog'] == 0:
        if latest_entry is not None:
            latest_sog = latest_entry.get('sog', 0)
            latest_nav_status = latest_entry.get('navigational_status', 15)

            if latest_sog == 0 or latest_nav_status == 1:
                should_add = False
                print("Skipping: Vessel not moving and latest entry also stopped")

    # Check time threshold
    if latest_entry_time is not None:
        try:
            report_time = parse_timestamp_with_tz(ship_data['timestamp_utc'])
        except:
            report_time = datetime.now(timezone.utc)

        if latest_entry_time.tzinfo is None:
            latest_entry_time = latest_entry_time.replace(tzinfo=timezone.utc)

        time_diff = report_time - latest_entry_time
        time_threshold = timedelta(hours=config.TIME_THRESHOLD_HOURS)

        if time_diff < time_threshold:
            should_add = False
            print(f"Skipping: Latest entry only {time_diff} old (< {config.TIME_THRESHOLD_HOURS} hour(s))")

    if should_add:
        # Append to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ship_data['timestamp_utc'],
                ship_data['mmsi'],
                ship_data['latitude'],
                ship_data['longitude'],
                ship_data['cog'],
                ship_data['sog'],
                ship_data['true_heading'],
                ship_data['nav_status'],
                weather_data.get('wetterzustand') if weather_data else None,
                weather_data.get('luftdruck') if weather_data else None,
                weather_data.get('windrichtung') if weather_data else None,
                weather_data.get('windstaerke') if weather_data else None,
                weather_data.get('bewoelkung') if weather_data else None,
                weather_data.get('lufttemperatur') if weather_data else None,
                weather_data.get('wassertemperatur') if weather_data else None,
                weather_data.get('niederschlag') if weather_data else None,
                weather_data.get('wellenhoehe') if weather_data else None
            ])

        print(f"✓ Added new entry to CSV: {csv_filename}")

        # Update latest entry
        new_latest_entry_time = parse_timestamp_with_tz(ship_data['timestamp_utc'])
        new_latest_entry = {
            'sog': ship_data['sog'],
            'navigational_status': ship_data['nav_status']
        }
    else:
        new_latest_entry_time = latest_entry_time
        new_latest_entry = latest_entry

    # Save complete data to JSON with full structure
    output = {
        "WeatherData": weather_data,
        "timestamp_utc": ship_data['timestamp_utc'],
        "PositionReport": {
            "Cog": ship_data['cog'],
            "CommunicationState": 999,
            "Latitude": ship_data['latitude'],
            "Longitude": ship_data['longitude'],
            "MessageID": 999,
            "NavigationalStatus": ship_data['nav_status'],
            "PositionAccuracy": True,
            "Raim": True,
            "RateOfTurn": 999,
            "RepeatIndicator": 999,
            "Sog": ship_data['sog'],
            "Spare": 999,
            "SpecialManoeuvreIndicator": 999,
            "Timestamp": 999,
            "TrueHeading": ship_data['true_heading'],
            "UserID": 999,
            "Valid": True
        },
        "MetaData": {
            "MMSI": str(ship_data['mmsi']),
            "MMSI_String": str(ship_data['mmsi']),
            "ShipName": ship_data['shipname'],
            "latitude": ship_data['latitude'],
            "longitude": ship_data['longitude'],
            "time_utc": ship_data['timestamp_utc']
        }
    }

    with open("position_report.json", "w") as f:
        json.dump(output, f, indent=2)

    return True, new_latest_entry_time, new_latest_entry, weather_data


# ============================================================================
# GEOJSON GENERATION
# ============================================================================

def create_ship_track_geojson(csv_file, output_file=None):
    """Convert AIS position reports CSV to GeoJSON track format (LineString)."""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['timestamp_utc'] = df['timestamp_utc'].apply(parse_timestamp_with_tz)
    df = df.sort_values('timestamp_utc')
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]

    geojson_features = []

    for mmsi, ship_data in df.groupby('mmsi'):
        if len(ship_data) < 2:
            continue

        coordinates = [[row['longitude'], row['latitude']] for _, row in ship_data.iterrows()]

        properties = {
            'mmsi': int(mmsi),
            'start_time': ship_data['timestamp_utc'].min().isoformat(),
            'end_time': ship_data['timestamp_utc'].max().isoformat(),
            'total_points': len(ship_data),
            'track_type': 'AIS_position_reports'
        }

        feature = {
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': coordinates},
            'properties': properties
        }

        geojson_features.append(feature)

    geojson = {'type': 'FeatureCollection', 'features': geojson_features}

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"GeoJSON track saved to {output_file}")

    return geojson


def create_ship_position_geojson(csv_file, output_file=None):
    """Convert AIS position reports CSV to GeoJSON point format (latest position per day)."""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'].apply(parse_timestamp_with_tz), utc=True)
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
    df['date'] = df['timestamp_utc'].dt.date

    geojson_features = []

    for mmsi, mmsi_data in df.groupby('mmsi'):
        daily_positions = []

        for date, day_data in mmsi_data.groupby('date'):
            latest_entry = day_data.loc[day_data['timestamp_utc'].idxmax()]
            daily_positions.append((date, latest_entry))

        daily_positions.sort(key=lambda x: x[0])

        # Filter positions within 500m
        filtered_positions = []
        for i, (date, entry) in enumerate(daily_positions):
            if i == 0:
                filtered_positions.append((date, entry))
            else:
                prev_date, prev_entry = filtered_positions[-1]
                current_pos = (entry['latitude'], entry['longitude'])
                prev_pos = (prev_entry['latitude'], prev_entry['longitude'])
                distance_m = geodesic(prev_pos, current_pos).meters

                if distance_m > 500:
                    filtered_positions.append((date, entry))

        for date, latest_entry in filtered_positions:
            properties = {
                'mmsi': int(mmsi),
                'date': date.strftime('%d.%m.%Y'),
                'timestamp_utc': latest_entry['timestamp_utc'].isoformat(),
                'cog': float(latest_entry['cog']) if pd.notna(latest_entry.get('cog')) else None,
                'sog': float(latest_entry['sog']) if pd.notna(latest_entry.get('sog')) else None,
                'navigational_status': int(latest_entry['navigational_status']) if pd.notna(latest_entry.get('navigational_status')) else None
            }

            weather_fields = ['wetterzustand', 'luftdruck', 'windrichtung', 'windstaerke',
                            'bewoelkung', 'lufttemperatur', 'wassertemperatur',
                            'niederschlag', 'wellenhoehe']

            for field in weather_fields:
                if field in latest_entry and pd.notna(latest_entry[field]):
                    value = latest_entry[field]
                    if hasattr(value, 'item'):
                        properties[field] = value.item()
                    else:
                        properties[field] = value

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [float(latest_entry['longitude']), float(latest_entry['latitude'])]
                },
                'properties': properties
            }

            geojson_features.append(feature)

    geojson = {'type': 'FeatureCollection', 'features': geojson_features}

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"GeoJSON positions saved to {output_file}")

    return geojson


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main execution function."""
    csv_filename = "ais_position_reports.csv"
    csv_headers = ["timestamp_utc", "mmsi", "latitude", "longitude", "cog", "sog",
                   "true_heading", "navigational_status", "wetterzustand", "luftdruck",
                   "windrichtung", "windstaerke", "bewoelkung", "lufttemperatur",
                   "wassertemperatur", "niederschlag", "wellenhoehe"]

    # Read latest entry from CSV
    latest_entry_time = None
    latest_entry = None

    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            if not df.empty:
                df['timestamp_utc'] = df['timestamp_utc'].apply(parse_timestamp_with_tz)
                latest_entry = df.loc[df['timestamp_utc'].idxmax()]
                latest_entry_time = latest_entry['timestamp_utc']
                print(f"Latest entry in CSV: {latest_entry_time}")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Create CSV if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        print(f"Created new CSV file: {csv_filename}")

    mmsi = config.FILTERS_SHIP_MMSI_ID[0]
    ship_id = config.FILTERS_SHIP_MMSI_ID[1]

    # OPTION 1: Try AIS Stream first
    print("=" * 60)
    print("OPTION 1: Attempting AIS Stream...")
    print("=" * 60)

    ship_data = await get_ship_data_from_aisstream(
        mmsi=mmsi,
        max_duration_minutes=config.MAX_DURATION_MINUTES,
        latest_entry_time=latest_entry_time
    )

    # OPTION 2: If AIS Stream fails, try scraping
    if ship_data is None:
        print("=" * 60)
        print("OPTION 2: Attempting Website Scraping...")
        print("=" * 60)

        ship_data = get_ship_data_from_scrapesite(
            ship_id=ship_id,
            mmsi=mmsi,
            debug=True,  # Enable debug output
            headless=False  # Show browser window
        )
    ship_data = None
    # OPTION 3: If scraping fails, try HAR analysis
    if ship_data is None:
        print("=" * 60)
        print("OPTION 3: Attempting HAR Analysis...")
        print("=" * 60)

        ship_data = get_ship_data_from_har_analysis(
            ship_id=ship_id,
            mmsi=mmsi,
            debug=True
        )
    # Process and save if we got data
    if ship_data:
        success, latest_entry_time, latest_entry, weather_data = await process_and_save_ship_data(
            ship_data=ship_data,
            latest_entry=latest_entry,
            latest_entry_time=latest_entry_time,
            csv_filename=csv_filename
        )

        # Generate GeoJSON outputs
        create_ship_track_geojson(csv_filename, 'ship_tracks.geojson')
        create_ship_position_geojson(csv_filename, 'ship_position.geojson')

        print("=" * 60)
        print("✓ SUCCESS - Data retrieved and saved")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ FAILURE - Could not retrieve ship data from any source")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
