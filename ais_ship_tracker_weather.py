import asyncio
import websockets
import json
import csv
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import aiohttp
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
            "FiltersShipMMSI": config.FILTERS_SHIP_MMSI,
            "FilterMessageTypes": ["PositionReport"]
        }

        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        start_time = datetime.now(timezone.utc)
        max_duration = timedelta(minutes=config.MAX_DURATION_MINUTES)
        found_result = False

        print(f"Starting AIS stream monitoring for {config.MAX_DURATION_MINUTES} minutes...")
        print(f"Monitoring MMSI(s): {config.FILTERS_SHIP_MMSI}")
        print(f"Time threshold: {config.TIME_THRESHOLD_HOURS} hour(s)")
        print(f"Start time: {start_time}")

        try:
            while True:
                current_time = datetime.now(timezone.utc)

                # Check if max duration has elapsed
                if current_time - start_time > max_duration:
                    print(f"{config.MAX_DURATION_MINUTES} minutes elapsed. Stopping stream.")
                    # Exit gracefully
                    print("Maximum duration reached. Exiting gracefully...")
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
                        # Parse the time_utc from the report (format: "2025-09-19 22:41:24.729345016 +0000 UTC")
                        try:
                            # Extract the datetime part and parse it
                            time_str = time_utc_from_report.replace(" +0000 UTC", "")
                            report_time = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
                        except:
                            # Fallback to current time if parsing fails
                            report_time = current_time

                        time_diff = report_time - latest_entry_time.replace(tzinfo=timezone.utc)
                        time_threshold = timedelta(hours=config.TIME_THRESHOLD_HOURS)  # Use config parameter

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
                            time_str = time_utc_from_report.replace(" +0000 UTC", "")
                            latest_entry_time = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
                        except:
                            latest_entry_time = current_time
                        latest_entry = {
                            'sog': sog,
                            'navigational_status': nav_status
                        }

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
    print(f"Monitored MMSI(s): {config.FILTERS_SHIP_MMSI}")
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