import asyncio
import websockets
import json
import csv
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time
import config  # Import configuration

def get_api_key():
    """
    Get API key from environment variable (GitHub Actions) or local secrets file
    """
    # First, try to get from environment variable (GitHub Actions)
    api_key = os.getenv('AISSTREAM_API_KEY')

    if api_key:
        return api_key

    # If not found in environment, try to load from local secrets file
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

def create_ship_track_geojson(csv_file, output_file=None):
    """
    Convert AIS position reports CSV to GeoJSON track format.

    Args:
        csv_file (str): Path to the AIS CSV file
        output_file (str): Output GeoJSON file path (optional)

    Returns:
        dict: GeoJSON object
    """

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # Sort by timestamp to ensure proper track order
    # Handle the specific datetime format with timezone
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], format='%Y-%m-%d %H:%M:%S.%f %z UTC', errors='coerce')
    df = df.sort_values('timestamp_utc')

    # Remove any rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]

    # Group by MMSI to handle multiple ships
    geojson_features = []

    for mmsi, ship_data in df.groupby('mmsi'):
        if len(ship_data) < 2:  # Skip if less than 2 points
            continue

        # Create coordinate array [longitude, latitude] for GeoJSON
        coordinates = []
        for _, row in ship_data.iterrows():
            coordinates.append([row['longitude'], row['latitude']])

        # Create properties with ship information
        properties = {
            'mmsi': int(mmsi),
            'start_time': ship_data['timestamp_utc'].min().isoformat(),
            'end_time': ship_data['timestamp_utc'].max().isoformat(),
            'total_points': len(ship_data),
            'track_type': 'AIS_position_reports'
        }

        # Add additional properties if available
        if 'navigational_status' in ship_data.columns:
            # Get the most common navigational status
            nav_status = ship_data['navigational_status'].mode().iloc[0] if not ship_data['navigational_status'].mode().empty else None
            if nav_status is not None:
                properties['navigational_status'] = int(nav_status)

        # Create LineString feature
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates
            },
            'properties': properties
        }

        geojson_features.append(feature)

    # Create the complete GeoJSON object
    geojson = {
        'type': 'FeatureCollection',
        'features': geojson_features
    }

    # Save to file if output path is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"GeoJSON saved to {output_file}")

    return geojson

async def get_weather_data(latitude, longitude):
    """
    Retrieve atmospheric and marine weather data from Open-Meteo API for given coordinates
    """
    try:
        # Get both atmospheric and marine weather data
        weather_data = {}

        # 1. Atmospheric weather data
        atmo_url = "https://api.open-meteo.com/v1/forecast"
        atmo_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "weather_code",           # Wetterzustand
                "surface_pressure",       # Luftdruck
                "wind_direction_10m",     # Windrichtung
                "wind_speed_10m",         # Windstärke
                "cloud_cover",            # Bewölkungsgrad
                "temperature_2m",         # Lufttemperatur
                "precipitation"           # Niederschlagsmenge
            ]
        }

        # 2. Marine weather data
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "sea_surface_temperature",  # Wassertemperatur
                "wave_height"               # Signifikante Wellenhöhe
            ]
        }

        async with aiohttp.ClientSession() as session:
            # Get atmospheric data
            async with session.get(atmo_url, params=atmo_params) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get('current', {})

                    # Map weather codes to German descriptions
                    weather_code = current.get('weather_code', None)
                    weather_description = get_weather_description(weather_code)

                    weather_data.update({
                        'wetterzustand': weather_description,
                        'luftdruck': current.get('surface_pressure', None),
                        'windrichtung': current.get('wind_direction_10m', None),
                        'windstaerke': current.get('wind_speed_10m', None),
                        'bewoelkung': current.get('cloud_cover', None),
                        'lufttemperatur': current.get('temperature_2m', None),
                        'niederschlag': current.get('precipitation', None)
                    })
                else:
                    print(f"Atmospheric weather API error: {response.status}")

            # Get marine data
            async with session.get(marine_url, params=marine_params) as response:
                if response.status == 200:
                    data = await response.json()
                    current = data.get('current', {})

                    weather_data.update({
                        'wassertemperatur': current.get('sea_surface_temperature', None),
                        'wellenhoehe': current.get('wave_height', None)
                    })
                else:
                    print(f"Marine weather API error: {response.status}")

        return weather_data if weather_data else None

    except Exception as e:
        print(f"Error retrieving weather data: {e}")
        return None

def get_weather_description(weather_code):
    """
    Convert WMO weather codes to German weather descriptions
    """
    if weather_code is None:
        return None

    weather_codes = {
        0: "Sonnig",
        1: "Überwiegend klar",
        2: "Teilweise bewölkt",
        3: "Bedeckt",
        45: "Nebel",
        48: "Nebel mit Reifablagerung",
        51: "Leichter Sprühregen",
        53: "Mäßiger Sprühregen",
        55: "Starker Sprühregen",
        56: "Leichter gefrierender Sprühregen",
        57: "Starker gefrierender Sprühregen",
        61: "Leichter Regen",
        63: "Mäßiger Regen",
        65: "Starker Regen",
        66: "Leichter gefrierender Regen",
        67: "Starker gefrierender Regen",
        71: "Leichter Schneefall",
        73: "Mäßiger Schneefall",
        75: "Starker Schneefall",
        77: "Schneekörner",
        80: "Leichte Regenschauer",
        81: "Mäßige Regenschauer",
        82: "Starke Regenschauer",
        85: "Leichte Schneeschauer",
        86: "Starke Schneeschauer",
        95: "Gewitter",
        96: "Gewitter mit leichtem Hagel",
        99: "Gewitter mit starkem Hagel"
    }

    return weather_codes.get(weather_code, f"Unbekannt ({weather_code})")

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
            'nav_status': 999,
            'timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.00000000 +0000 UTC'),
            'sog': 999,
            'cog': 999,
            'shipname': 'none'
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

            # Extract ship name
            elif 'Name' in header:
                result['shipname'] = value
                if debug:
                    print(f"  → shipname = {result['shipname']}")


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
        'cog': details['cog'],
        'shipname': details['shipname']
    }

    return result

async def process_and_save_ship_data(
    timestamp_utc,
    mmsi,
    latitude,
    longitude,
    cog,
    sog,
    true_heading,
    nav_status,
    latest_entry,
    latest_entry_time,
    csv_filename,
    shipname=None,
    ais_message=None,
    meta_data=None
):
    """
    Process ship position data, check filtering rules, get weather data, and save to CSV.

    Args:
        timestamp_utc: UTC timestamp string
        mmsi: Maritime Mobile Service Identity
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        cog: Course over ground
        sog: Speed over ground
        true_heading: True heading
        nav_status: Navigational status
        latest_entry: Dictionary with latest entry data (sog, navigational_status)
        latest_entry_time: Datetime of latest entry
        csv_filename: Path to CSV file
        ais_message: Optional AIS message dict for JSON output
        meta_data: Optional metadata dict for JSON output

    Returns:
        Tuple of (should_continue, new_latest_entry_time, new_latest_entry)
        - should_continue: Boolean indicating if processing was successful
        - new_latest_entry_time: Updated latest entry time
        - new_latest_entry: Updated latest entry dict
    """
    print(f"[{timestamp_utc}] MMSI: {mmsi}, Lat: {latitude}, Lon: {longitude}, SOG: {sog}, Nav Status: {nav_status}")

    # Get weather data for the ship's current position
    print("Retrieving weather data...")
    weather_data = await get_weather_data(latitude, longitude)

    if weather_data:
        print(f"Wetter: {weather_data.get('wetterzustand', 'N/A')}, Temp: {weather_data.get('lufttemperatur', 'N/A')}°C, Wind: {weather_data.get('windstaerke', 'N/A')} km/h")

    # Check if we should add this entry based on the filtering rules
    should_add = True

    # Rule: Don't add if current vessel is NOT moving (nav_status = 1 OR sog = 0)
    # AND latest entry also has sog == 0 OR nav_status == 1
    if nav_status == 1 or sog == 0:  # Current vessel is not moving
        if latest_entry is not None:
            latest_sog = latest_entry.get('sog', 0)
            latest_nav_status = latest_entry.get('navigational_status', 15)

            if latest_sog == 0 or latest_nav_status == 1:
                should_add = False
                print("Skipping entry: Current vessel not moving and latest entry was also stopped/anchored")

    # Check if latest entry is older than configured threshold
    if latest_entry_time is not None:
        # Parse the time_utc from the report
        try:
            time_str = timestamp_utc.replace(" +0000 UTC", "")
            report_time = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
        except:
            # Fallback to current time if parsing fails
            report_time = datetime.now(timezone.utc)

        time_diff = report_time - latest_entry_time.replace(tzinfo=timezone.utc)
        time_threshold = timedelta(hours=config.TIME_THRESHOLD_HOURS)

        if time_diff < time_threshold:
            should_add = False
            print(f"Skipping entry: Latest entry is only {time_diff} old (< {config.TIME_THRESHOLD_HOURS} hour(s))")

    if should_add:
        # Append to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_utc,
                mmsi,
                latitude,
                longitude,
                cog,
                sog,
                true_heading,
                nav_status,
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

        print(f"Added new entry to CSV: {csv_filename}")

        # Update latest entry info for next iteration
        try:
            time_str = timestamp_utc.replace(" +0000 UTC", "")
            new_latest_entry_time = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
        except:
            new_latest_entry_time = datetime.now(timezone.utc)

        new_latest_entry = {
            'sog': sog,
            'navigational_status': nav_status
        }
    else:
        # Don't update if we didn't add
        new_latest_entry_time = latest_entry_time
        new_latest_entry = latest_entry

    # Save the complete message to JSON for reference
    output = {
        "WeatherData": weather_data,
        "timestamp_utc": timestamp_utc
    }

    if ais_message is not None:
        output["PositionReport"] = ais_message
    else:
        # For fallback data - create structure matching default mode
        output["PositionReport"] = {
            "Cog": cog,
            "CommunicationState": 999,
            "Latitude": latitude,
            "Longitude": longitude,
            "MessageID": 999,
            "NavigationalStatus": nav_status,
            "PositionAccuracy": True,
            "Raim": True,
            "RateOfTurn": 999,
            "RepeatIndicator": 999,
            "Sog": sog,
            "Spare": 999,
            "SpecialManoeuvreIndicator": 999,
            "Timestamp": 999,
            "TrueHeading": cog,
            "UserID": 999,
            "Valid": True
        }

    if meta_data is not None:
        output["MetaData"] = meta_data
    else:
        # For fallback data - create MetaData structure
        output["MetaData"] = {
            "MMSI": mmsi,
            "MMSI_String": mmsi,
            "ShipName": shipname,
            "latitude": latitude,
            "longitude": longitude,
            "time_utc": timestamp_utc
        }

    with open("position_report.json", "w") as f:
        json.dump(output, f, indent=2)

    return True, new_latest_entry_time, new_latest_entry

async def connect_ais_stream():
    csv_filename = "ais_position_reports.csv"
    csv_headers = ["timestamp_utc", "mmsi", "latitude", "longitude", "cog", "sog", "true_heading", "navigational_status","wetterzustand", "luftdruck", "windrichtung", "windstaerke", "bewoelkung", "lufttemperatur","wassertemperatur", "niederschlag", "wellenhoehe"]

    # Check if CSV exists and get the latest entry
    latest_entry_time = None
    latest_entry = None

    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            if not df.empty:
                # Get the most recent entry
                # Parse timestamps with custom format handling
                def parse_timestamp(ts):
                    try:
                        # Handle format: "2025-09-20 15:57:15.789483127 +0000 UTC"
                        time_str = str(ts).replace(" +0000 UTC", "")
                        return pd.to_datetime(time_str, utc=True)
                    except:
                        return pd.to_datetime(ts, utc=True)

                df['timestamp_utc'] = df['timestamp_utc'].apply(parse_timestamp)
                latest_entry = df.loc[df['timestamp_utc'].idxmax()]
                latest_entry_time = latest_entry['timestamp_utc']
                print(f"Latest entry in CSV: {latest_entry_time}")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Create CSV with headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        print(f"Created new CSV file: {csv_filename}")

    # get API key
    api_key = get_api_key()
    if not api_key:
        print("Error: API key not found. Please set AISSTREAM_API_KEY environment variable or create secrets/aisstream.json")
        return

    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": api_key,
            #"BoundingBoxes": [[[-90, -180], [90, 180]]], # Worldwide
            "BoundingBoxes": config.BOUNDING_BOXES,
            "FiltersShipMMSI": [config.FILTERS_SHIP_MMSI_ID[0]],
            "FilterMessageTypes": ["PositionReport"]
        }

        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        start_time = datetime.now(timezone.utc)
        max_duration = timedelta(minutes=config.MAX_DURATION_MINUTES)
        found_result = False

        print(f"Starting AIS stream monitoring for {config.MAX_DURATION_MINUTES} minutes...")
        print(f"Monitoring MMSI(s): {config.FILTERS_SHIP_MMSI_ID[0]}")
        print(f"Time threshold: {config.TIME_THRESHOLD_HOURS} hour(s)")
        print(f"Start time: {start_time}")


        try:
            while True:
                current_time = datetime.now(timezone.utc)

                # Check if max duration has elapsed
                if current_time - start_time > max_duration:
                    print(f"{config.MAX_DURATION_MINUTES} minutes elapsed. Stopping stream.")
                    # Exit gracefully
                    print("Maximum duration reached. Going for fallback solution...")

                    # Extract all data
                    data = get_ship_data(config.FILTERS_SHIP_MMSI_ID[1], debug=False)

                    if data:
                        found_result = True
                        print("\n" + "=" * 60)
                        print("✓ FALLBACK SUCCESS - Complete Ship Data:")
                        time_utc_from_report = data['timestamp_utc']
                        timestamp_utc = time_utc_from_report # Use current time as timestamp
                        mmsi = config.FILTERS_SHIP_MMSI_ID[0]
                        latitude = data['lat']
                        longitude = data['lon']
                        cog = data['cog']
                        sog = data['sog']
                        true_heading = 511
                        nav_status = data['nav_status']
                        shipname=data['shipname']
                        print("=" * 60)
                        print(f"[{timestamp_utc}] MMSI: {mmsi}, Lat: {latitude}, Lon: {longitude}, SOG: {sog}, Nav Status: {nav_status}")
                        success, latest_entry_time, latest_entry = await process_and_save_ship_data(
                            timestamp_utc=timestamp_utc,
                            mmsi=mmsi,
                            latitude=latitude,
                            longitude=longitude,
                            cog=cog,
                            sog=sog,
                            true_heading=true_heading,
                            nav_status=nav_status,
                            latest_entry=latest_entry,
                            latest_entry_time=latest_entry_time,
                            csv_filename=csv_filename,
                            shipname=shipname
                        )




                    else:
                        print("\n✗ FAILED - Could not extract ship data")
                        print("=" * 60)
                        print("Also fallback solution failed...Exciting gracefully")
                    return
                try:
                    message_json = await asyncio.wait_for(websocket.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue  # Check time again

                message = json.loads(message_json)
                message_type = message["MessageType"]

                if message_type == "PositionReport":
                    found_result = True
                    ais_message = message['Message']['PositionReport']
                    meta_data = message['MetaData']

                    # Extract position report data
                    # Use the time_utc from MetaData instead of current_time
                    time_utc_from_report = meta_data.get('time_utc', current_time.isoformat())
                    timestamp_utc = time_utc_from_report
                    mmsi = ais_message.get('UserID', '')
                    latitude = round(ais_message.get('Latitude', 0), 5)
                    longitude = round(ais_message.get('Longitude', 0),5)
                    cog = ais_message.get('Cog', 360)  # 360 = not available
                    sog = ais_message.get('Sog', 0)   # Speed over ground (knots * 10)
                    true_heading = ais_message.get('TrueHeading', 511)  # 511 = not available
                    nav_status = ais_message.get('NavigationalStatus', 15)  # 15 = not defined

                    print(f"[{timestamp_utc}] MMSI: {mmsi}, Lat: {latitude}, Lon: {longitude}, SOG: {sog}, Nav Status: {nav_status}")
                    success, latest_entry_time, latest_entry = await process_and_save_ship_data(
                        timestamp_utc=timestamp_utc,
                        mmsi=mmsi,
                        latitude=latitude,
                        longitude=longitude,
                        cog=cog,
                        sog=sog,
                        true_heading=true_heading,
                        nav_status=nav_status,
                        latest_entry=latest_entry,
                        latest_entry_time=latest_entry_time,
                        csv_filename=csv_filename,
                        ais_message=ais_message,
                        meta_data=meta_data
                    )
                    # Also save the complete message to JSON for reference and as well the weather data
                    output = {
                        "PositionReport": ais_message,
                        "MetaData": meta_data,
                        "WeatherData": weather_data,
                        "timestamp_utc": timestamp_utc
                    }

                    with open("position_report.json", "w") as f:
                        json.dump(output, f, indent=2)

                    # Stop after processing the first message for our MMSI
                    print("First AIS message received and processed. Stopping stream.")
                    break

        except asyncio.TimeoutError:
            print("WebSocket timeout occurred")
        except Exception as e:
            print(f"Error during streaming: {e}")

        if not found_result:
            print(f"No AIS messages received for the specified MMSI(s) within {config.MAX_DURATION_MINUTES} minutes.")
            print("Existing files were not overwritten.")
        else:
            print(f"AIS monitoring completed. Data saved to {csv_filename}")

def print_field_meanings():
    """Print the field meanings for reference"""
    print("\n=== Configuration ===")
    print(f"Monitored MMSI(s): {config.FILTERS_SHIP_MMSI_ID[0]}")
    print(f"Max monitoring duration: {config.MAX_DURATION_MINUTES} minutes")
    print(f"Time threshold for duplicate filtering: {config.TIME_THRESHOLD_HOURS} hour(s)")

    print("\n=== AIS PositionReport Field Meanings ===")
    print("1) Cog (Course over Ground): Vessel's actual course over ground in degrees (0-359). 360 = not available")
    print("2) NavigationalStatus: 0=Under way, 1=At anchor, 2=Not under command, 8=Reserved, 15=Not defined")
    print("3) Latitude and Longitude: GPS coordinates in decimal degrees")
    print("4) Sog (Speed over Ground): Vessel speed in knots * 10. 0 = stopped")
    print("5) TrueHeading: Heading from ship's sensor in degrees (0-359). 511 = not available")
    print("6) MMSI: Maritime Mobile Service Identity - unique vessel identifier")
    print("\n=== Weather Data Fields ===")
    print("7) Wetterzustand: Weather condition (Sonnig, Nebel, Gewitter, etc.)")
    print("8) Luftdruck: Atmospheric pressure in hPa")
    print("9) Windrichtung: Wind direction in degrees (0-360)")
    print("10) Windstärke: Wind speed in km/h")
    print("11) Bewölkung: Cloud cover percentage (0-100%)")
    print("12) Lufttemperatur: Air temperature in °C")
    print("13) Wassertemperatur: Sea surface temperature in °C")
    print("14) Niederschlag: Precipitation amount in mm")
    print("15) Wellenhöhe: Significant wave height in meters")
    print("=" * 60)

if __name__ == "__main__":
    #print_field_meanings()
    #main asyncio loop
    asyncio.run(connect_ais_stream())
    # Generate track GeoJSON (LineString)
    track_geojson = create_ship_track_geojson('ais_position_reports.csv', 'ship_tracks.geojson')