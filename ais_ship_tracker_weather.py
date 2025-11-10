import asyncio
from unittest import result
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
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time
import config  # Import configuration
from dateutil import parser as date_parser
import undetected_chromedriver as uc




import math  # Add to your imports if not already there



def create_bounding_box_around_position(latitude, longitude, radius_km=200):
    """
    Create a bounding box around a given position.

    Args:
        latitude: Center latitude in degrees
        longitude: Center longitude in degrees
        radius_km: Radius around the center point in kilometers (default: 200)

    Returns:
        Bounding box in format [[[lat_min, lon_min], [lat_max, lon_max]]]
        Compatible with AIS Stream API
    """
    # Convert radius to degrees
    # Latitude: 1 degree ≈ 111 km
    lat_delta = radius_km / 111.0

    # Longitude: depends on latitude (cos(lat) correction)
    # At equator: 1 degree ≈ 111 km, at poles: 0 km
    lon_delta = radius_km / (111.0 * math.cos(math.radians(latitude)))

    # Calculate bounds
    lat_min = max(-90, latitude - lat_delta)
    lat_max = min(90, latitude + lat_delta)
    lon_min = max(-180, longitude - lon_delta)
    lon_max = min(180, longitude + lon_delta)

    # Return in AIS Stream API format
    return [[[lat_min, lon_min], [lat_max, lon_max]]]

def parse_timestamp_with_tz(timestamp_str):
    """
    Parse timestamp string preserving timezone information.
    Handles multiple formats including ISO format and custom AIS formats with nanoseconds.
    Truncates to whole seconds (no fractional seconds).
    """
    try:
        # Handle AIS Stream format with nanoseconds: "2025-11-05 13:59:49.876462847 +0000 UTC"
        # Remove fractional seconds entirely
        if ' UTC' in timestamp_str and '+' in timestamp_str:
            # Remove the " UTC" suffix
            timestamp_str = timestamp_str.replace(' UTC', '')

            # Split into datetime part and timezone part
            parts = timestamp_str.rsplit('+', 1)
            if len(parts) == 2:
                dt_part = parts[0].strip()
                tz_part = '+' + parts[1].strip()

                # Remove fractional seconds completely
                if '.' in dt_part:
                    dt_part = dt_part.split('.')[0]

                # Reconstruct the timestamp
                timestamp_str = f"{dt_part}{tz_part}"

        # Try dateutil parser first
        parsed_dt = date_parser.parse(timestamp_str)
        # Remove microseconds from the parsed datetime
        return parsed_dt.replace(microsecond=0)
    except:
        try:
            # Try standard ISO format
            parsed_dt = datetime.fromisoformat(timestamp_str)
            return parsed_dt.replace(microsecond=0)
        except:
            try:
                # Try parsing without timezone and add UTC
                dt = datetime.strptime(timestamp_str.split('+')[0].strip().split('.')[0], '%Y-%m-%d %H:%M:%S')
                return dt.replace(tzinfo=timezone.utc, microsecond=0)
            except:
                print(f"Warning: Could not parse timestamp '{timestamp_str}', using current UTC time")
                return datetime.now(timezone.utc).replace(microsecond=0)

def format_timestamp_with_tz(dt):
    """
    Format datetime object to ISO 8601 string preserving timezone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


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
    df['timestamp_utc'] = df['timestamp_utc'].apply(parse_timestamp_with_tz)
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

def create_ship_position_geojson(csv_file, output_file=None):
    """
    Convert AIS position reports CSV to GeoJSON point format with latest position per day.

    Args:
        csv_file (str): Path to the AIS CSV file
        output_file (str): Output GeoJSON file path (optional)

    Returns:
        dict: GeoJSON object with Point features for latest position each day
    """

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # Parse timestamps - keep as pandas datetime for proper .dt accessor support
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'].apply(parse_timestamp_with_tz), utc=True)

    # Remove any rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]

    # Extract date (without time) for grouping - now .dt accessor will work
    df['date'] = df['timestamp_utc'].dt.date

    # Group by MMSI and date, get the latest position for each day
    geojson_features = []

    for mmsi, mmsi_data in df.groupby('mmsi'):
        for date, day_data in mmsi_data.groupby('date'):
            # Get the latest entry for this day
            latest_entry = day_data.loc[day_data['timestamp_utc'].idxmax()]

            # Format date as DD.MM.YYYY
            date_formatted = date.strftime('%d.%m.%Y')

            # Create properties with ship and position information
            properties = {
                'mmsi': int(mmsi),
                'date': date_formatted,
                'timestamp_utc': latest_entry['timestamp_utc'].isoformat(),
                'cog': float(latest_entry['cog']) if pd.notna(latest_entry.get('cog')) else None,
                'sog': float(latest_entry['sog']) if pd.notna(latest_entry.get('sog')) else None,
                'navigational_status': int(latest_entry['navigational_status']) if pd.notna(latest_entry.get('navigational_status')) else None
            }

            # Add weather data if available
            weather_fields = ['wetterzustand', 'luftdruck', 'windrichtung', 'windstaerke',
                            'bewoelkung', 'lufttemperatur', 'wassertemperatur',
                            'niederschlag', 'wellenhoehe']

            for field in weather_fields:
                if field in latest_entry and pd.notna(latest_entry[field]):
                    value = latest_entry[field]
                    # Convert numpy types to native Python types
                    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)) or hasattr(value, 'item'):
                        properties[field] = value.item() if hasattr(value, 'item') else value
                    elif isinstance(value, str):
                        properties[field] = value
                    else:
                        # Handle numeric types
                        try:
                            properties[field] = float(value) if '.' in str(value) else int(value)
                        except (ValueError, TypeError):
                            properties[field] = str(value)

            # Create Point feature with explicit type conversion for coordinates
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [float(latest_entry['longitude']), float(latest_entry['latitude'])]
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
    """
    Sets up the Selenium WebDriver using undetected-chromedriver.
    """
    options = uc.ChromeOptions()
    headless = False
    if headless:
        options.add_argument('--headless=new') # Use the new headless mode

    # Add a realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36')

    # Other useful options
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled') # Disables navigator.webdriver

    # Initialize undetected-chromedriver
    # You might need to specify a driver_executable_path
    # driver = uc.Chrome(options=options, driver_executable_path='/path/to/chromedriver')
    driver = uc.Chrome(options=options)

    return driver



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
    Extract coordinates by analyzing the page's JavaScript (Method 1),
    direct API access (Method 2).

    Looks for map initialization code with retry logic.
    """
    # Assuming 'setup_driver' is defined and returns a Selenium WebDriver instance
    driver = setup_driver(headless=not debug)

    debug = True

    try:
        if debug:
            print(f"Loading URL: {url}")

        driver.get(url)
        # Assuming 'handle_cookie_consent' is defined
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
                # Even if no map container, we'll continue to Method 2 and 3,
                # as they don't strictly require the map DOM element.
                pass

        time.sleep(3)

        # --- Method 1: JavaScript/Leaflet Object Access (Primary Method) ---
        for attempt in range(max_retries):
            try:
                if debug:
                    print(f"Attempt {attempt + 1}/{max_retries} for Method 1 (JS Access)...")

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
                        print(f"✓ Found coordinates via Method 1 (JS): {center.get('method', 'unknown')}")
                    return {
                        'lat': float(center['lat']),
                        'lon': float(center['lng'])
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


        # -------------------------------------------------------------------
        # --- Method 2: Direct API Navigation (New Fallback) ---
        # -------------------------------------------------------------------
        coordinates_from_api = None
        if debug:
            print("Method 1 (JS) failed. Attempting Method 2: Direct API Navigation (MarineTraffic).")

        try:
            # Wait for API call to complete (Crucial for selenium-wire)
            if debug:
                print("  Waiting 10s for API calls to complete...")
            time.sleep(10)

            # 1. Extract the ship ID from the original URL
            target_ship_id = None
            id_match = re.search(r'shipid:(\d+)', url)
            if id_match:
                target_ship_id = id_match.group(1)

            if not target_ship_id:
                if debug:
                    print("  Failed to find shipid in URL for Method 2. Skipping API call.")
            else:
                api_url = f"https://www.marinetraffic.com/map/getvesseljson/shipid:{target_ship_id}"

                if debug:
                    print(f"  Attempting to navigate to API URL: {api_url}")

                # 2. Navigate to the API endpoint, leveraging the authenticated session
                driver.get(api_url)

                # Wait briefly for content to load
                time.sleep(1)

                # 3. Read the page source (which should be raw JSON)
                raw_json_data = driver.page_source

                # Clean up the JSON text by removing potential surrounding HTML (e.g., <pre> tags)
                clean_json_text = re.sub(r'<[^>]*>', '', raw_json_data, flags=re.IGNORECASE).strip()

                if not clean_json_text.startswith('{'):
                    if debug:
                        print(f"  Content does not look like raw JSON: {clean_json_text[:50]}...")
                else:
                    # 4. Parse the JSON
                    data = json.loads(clean_json_text)

                    # 5. Extract coordinates
                    if 'LAT' in data and 'LON' in data:
                        lat = float(data['LAT'])
                        lon = float(data['LON'])
                        if debug:
                            print(f"✓ COORDINATES EXTRACTED via Method 2: Direct API Navigation: lat={lat}, lon={lon}")
                        coordinates_from_api = {'lat': lat, 'lon': lon}
                    else:
                        if debug:
                            print("  ✗ JSON found, but 'LAT'/'LON' keys are missing.")

        except Exception as e:
            if debug:
                print(f"  Method 2 failed: {e}")

        # Return coordinates if Method 2 was successful
        if coordinates_from_api:
            return coordinates_from_api


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
    #debug = True

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
            'timestamp_utc': format_timestamp_with_tz(datetime.now(timezone.utc)),
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
                tz_match = re.search(r'UTC([+-]\d+)', value)
                if time_match:
                    time_str = time_match.group(1)
                    try:
                        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
                        # Apply timezone offset if found
                        if tz_match:
                            offset_hours = int(tz_match.group(1))
                            tz = timezone(timedelta(hours=offset_hours))
                            dt = dt.replace(tzinfo=tz)
                        else:
                            dt = dt.replace(tzinfo=timezone.utc)
                        result['timestamp_utc'] = format_timestamp_with_tz(dt)
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

    Returns:
        Tuple of (should_continue, new_latest_entry_time, new_latest_entry, weather_data)
        - should_continue: Boolean indicating if processing was successful
        - new_latest_entry_time: Updated latest entry time
        - new_latest_entry: Updated latest entry dict
        - weather_data: Weather data dictionary
    """
    #print(f"[{timestamp_utc}] MMSI: {mmsi}, Lat: {latitude}, Lon: {longitude}, SOG: {sog}, Nav Status: {nav_status}")

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
            report_time = parse_timestamp_with_tz(timestamp_utc)
        except:
            report_time = datetime.now(timezone.utc)

        # Ensure both timestamps are timezone-aware for comparison
        if latest_entry_time.tzinfo is None:
            latest_entry_time = latest_entry_time.replace(tzinfo=timezone.utc)
        time_diff = report_time - latest_entry_time
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
            new_latest_entry_time = parse_timestamp_with_tz(timestamp_utc)
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
        # Create a copy of meta_data and convert time_utc timestamp
        meta_data_copy = meta_data.copy()
        if 'time_utc' in meta_data_copy:
            parsed_time = parse_timestamp_with_tz(meta_data_copy['time_utc'])
            meta_data_copy['time_utc'] = format_timestamp_with_tz(parsed_time)
        output["MetaData"] = meta_data_copy
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

    # Return weather_data as the 4th element
    return True, new_latest_entry_time, new_latest_entry, weather_data

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
                        return parse_timestamp_with_tz(str(ts))
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
    # Create bounding box around latest known position

    bounding_box = create_bounding_box_around_position(float(latest_entry['latitude']), float(latest_entry['longitude']), radius_km=200)
    # Connect to AIS WebSocket stream
    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": api_key,
            #"BoundingBoxes": [[[-90, -180], [90, 180]]], # Worldwide
            "BoundingBoxes": config.BOUNDING_BOXES, #use config bounding boxes
            #"BoundingBoxes": [bounding_box[0]],
            #"BoundingBoxes": [[[47.8369281981982, -4.40158439920577], [51.440531801801804, 1.1629243992057703]]], # testing
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
                        parsed_time = parse_timestamp_with_tz(time_utc_from_report)
                        timestamp_utc = format_timestamp_with_tz(parsed_time)
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
                        success, latest_entry_time, latest_entry, weather_data = await process_and_save_ship_data(
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
                    # Parse and reformat the timestamp to standardized format
                    parsed_time = parse_timestamp_with_tz(time_utc_from_report)
                    timestamp_utc = format_timestamp_with_tz(parsed_time)
                    mmsi = ais_message.get('UserID', '')
                    latitude = round(ais_message.get('Latitude', 0), 5)
                    longitude = round(ais_message.get('Longitude', 0),5)
                    cog = ais_message.get('Cog', 360)  # 360 = not available
                    sog = ais_message.get('Sog', 0)   # Speed over ground (knots * 10)
                    true_heading = ais_message.get('TrueHeading', 511)  # 511 = not available
                    nav_status = ais_message.get('NavigationalStatus', 15)  # 15 = not defined

                    print(f"[{timestamp_utc}] MMSI: {mmsi}, Lat: {latitude}, Lon: {longitude}, SOG: {sog}, Nav Status: {nav_status}")
                    success, latest_entry_time, latest_entry, weather_data = await process_and_save_ship_data(
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
                    # Create a copy of meta_data and convert time_utc timestamp
                    meta_data_copy = meta_data.copy()
                    if 'time_utc' in meta_data_copy:
                        parsed_time = parse_timestamp_with_tz(meta_data_copy['time_utc'])
                        meta_data_copy['time_utc'] = format_timestamp_with_tz(parsed_time)
                    output = {
                        "PositionReport": ais_message,
                        "MetaData": meta_data_copy,
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
    try:
        #print_field_meanings()
        #main asyncio loop
        asyncio.run(connect_ais_stream())
        # Generate track GeoJSON (LineString)
        track_geojson = create_ship_track_geojson('ais_position_reports.csv', 'ship_tracks.geojson')
        # Generate position points GeoJSON (Point features)
        position_geojson = create_ship_position_geojson('ais_position_reports.csv', 'ship_position.geojson')
        
        # Explicit success exit
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
