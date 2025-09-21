
# ShipTracker

ShipTracker is a Python-based application for collecting, enriching, and visualizing real-time and historical ship positions and weather data. It uses AIS (Automatic Identification System) position reports, integrates weather information from Open-Meteo, and displays ship tracks and live status on an interactive Leaflet map.

See the [DEMO LIVETRACKING](https://davidoesch.github.io/shiptracker/index.html)

## Features

- Collects AIS position reports for specified ships (MMSI filter)
- Enriches AIS data with current weather and marine conditions
- Stores data in CSV and JSON formats
- Generates GeoJSON tracks for map visualization
- Interactive web map (Leaflet) for viewing ship position, track, and weather
- Configurable monitoring duration and duplicate filtering

## Project Structure

- `ais_ship_tracker_weather.py` — Main Python script for data collection and enrichment
- `config.py` — Configuration for ship MMSI, monitoring duration, and filtering
- `ais_position_reports.csv` — Collected AIS and weather data
- `ship_tracks.geojson` — Generated ship track data for map visualization
- `position_report.json` — Latest ship position and weather data
- `index.html` — Interactive map visualization (Leaflet)
- `secrets/aisstream.json` — Local secrets file for AIS API key (not tracked in git)
- `.github/workflows/run_ais.yml` — GitHub Actions workflow for automation

## Quick Start

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Set up AIS API key:**
- Create a free account at [AISstream](https://aisstream.io/accounts/register)
- Add your API key to `secrets/aisstream.json` (you can find it in your AISstream account):
     ```json
     { "APIKey": "your_api_key_here" }
     ```

3. **Configure ship(s) to monitor:**
   - Edit `config.py` and set `FILTERS_SHIP_MMSI` to your target MMSI(s).

4. **Run the tracker:**
   ```sh
   python ais_ship_tracker_weather.py
   ```

5. **View results:**
   - Open `index.html` in your browser to view the map and ship data.

## Data Files

- **CSV:** All collected AIS and weather data (`ais_position_reports.csv`)
- **GeoJSON:** Ship tracks for visualization (`ship_tracks.geojson`)
- **JSON:** Latest ship position and weather (`position_report.json`)

## Configuration

See [`config.py`](config.py) for all available options:
- `FILTERS_SHIP_MMSI`: List of MMSI numbers to monitor
- `MAX_DURATION_MINUTES`: How long to monitor the AIS stream
- `TIME_THRESHOLD_HOURS`: Duplicate filtering threshold

## Demo Website & Permalink Usage

You can view the ShipTracker demo at  
**[https://davidoesch.github.io/shiptracker/index.html](https://davidoesch.github.io/shiptracker/index.html)**

The map supports permalinks via URL parameters:

- `?zoom=14`  
  Sets the initial zoom level of the map (default is 12).

- `?popupzoom=false`  
  Prevents the ship info popup from opening automatically when the page loads (default is `true`).

**Examples:**

- [Show map with zoom 14](https://davidoesch.github.io/shiptracker/index.html?zoom=14)
- [Show map with zoom 14 and no popup](https://davidoesch.github.io/shiptracker/index.html?zoom=14&popupzoom=false)
- [Show map with no popup](https://davidoesch.github.io/shiptracker/index.html?popupzoom=false)

Just append the desired parameters to the URL to customize your view and share direct links to specific map states.

## License

This project is licensed under the GNU GPL v3. See [`LICENSE`](LICENSE) for details.

---

For technical details, see the [Technical Description](https://deepwiki.com/davidoesch/shiptracker/1-overview).


