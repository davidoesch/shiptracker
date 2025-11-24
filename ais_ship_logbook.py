"""
Nautical Logbook PDF Generator
Generates daily logbook entries from AIS position reports CSV with charts and maps
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from datetime import datetime, timedelta
from io import BytesIO
import requests
import json
import locale
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                Paragraph, PageBreak, Spacer, Image as RLImage)
from reportlab.graphics.shapes import Drawing, Circle, Line, Wedge
import math
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import PageTemplate, Frame, BaseDocTemplate
from reportlab.pdfgen import canvas
import os
from PyPDF2 import PdfReader, PdfWriter
import re
import locale

try:
    locale.setlocale(locale.LC_TIME, "de_CH.UTF-8")
except locale.Error:
    locale.setlocale(locale.LC_TIME, "de_CH.utf8")

try:
    from config import VESSEL_INFO, LOGBOOK_SETTINGS
    HAS_CONFIG = True
except ImportError:
    print("Warning: conf.py not found. Using default vessel information.")
    HAS_CONFIG = False
    VESSEL_INFO = {
        'name': 'Halleluja',
        'prefix': 'M/V',
        'flag_state': 'Unknown',
        'vessel_type': 'Unknown'
    }
    LOGBOOK_SETTINGS = {
        'title': 'NAUTICAL LOGBOOK',
        'subtitle': 'Voyage Log',
        'show_flag_state': False,
        'show_vessel_type': False
    }

class NumberedCanvas(canvas.Canvas):
    """Custom canvas to add page numbers and footer."""

    def __init__(self, *args, **kwargs):
        # Extract page_offset if provided
        self.page_offset = kwargs.pop('page_offset', 0)
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page numbers and footers to all pages."""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number_and_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number_and_footer(self, page_count):
        """Draw footer with page numbers (skip first 2 pages)."""
        page_num = self._pageNumber

        # Skip title page (page 1) and track overview page (page 2)
        if page_num <= 2:
            return

        # Set font for footer
        self.setFont('Helvetica', 9)
        self.setFillColorRGB(0.5, 0.5, 0.5)

        # Get page dimensions (landscape A4)
        page_width = landscape(A4)[0]

        # Left side: Document title
        vessel_name = f"{VESSEL_INFO.get('prefix', 'M/V')} {VESSEL_INFO.get('name', 'Unknown')}"
        self.drawString(15*mm, 8*mm, f"{LOGBOOK_SETTINGS.get('title', 'Nautical Logbook')} - {vessel_name}")

        # Center: Date/time generated
        from datetime import datetime
        now = datetime.now().strftime('%d.%m.%Y')
        center_text = f"Erstellt: {now}"
        text_width = self.stringWidth(center_text, 'Helvetica', 9)
        self.drawString((page_width - text_width) / 2, 8*mm, center_text)

        # Right side: Page numbers (adjusted with offset)
        # Apply page_offset to continue numbering from existing PDF
        adjusted_page_num = page_num - 2 + self.page_offset
        adjusted_total = page_count - 2 + self.page_offset

        page_text = f"Seite {adjusted_page_num} "
        text_width = self.stringWidth(page_text, 'Helvetica', 9)
        self.drawString(page_width - 15*mm - text_width, 8*mm, page_text)

        # Draw a thin line above the footer
        self.setStrokeColorRGB(0.7, 0.7, 0.7)
        self.setLineWidth(0.5)
        self.line(15*mm, 12*mm, page_width - 15*mm, 12*mm)

def make_canvas_with_offset(page_offset):
    """Factory function to create NumberedCanvas with page offset."""
    class OffsetCanvas(NumberedCanvas):
        def __init__(self, *args, **kwargs):
            kwargs['page_offset'] = page_offset
            super().__init__(*args, **kwargs)
    return OffsetCanvas

class FooteredDocTemplate(BaseDocTemplate):
    """Custom document template with footer on all pages except first two."""

    def __init__(self, filename, **kwargs):
        BaseDocTemplate.__init__(self, filename, **kwargs)

        # Define frame for content (same as SimpleDocTemplate would use)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id='normal'
        )

        # Create page templates
        template = PageTemplate(id='Later', frames=[frame])
        self.addPageTemplates([template])

# ============================================================================
# CONSTANTS
# ============================================================================

EARTH_RADIUS_NM = 3440.065  # Earth radius in nautical miles
WIND_DIRECTIONS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# Drawing dimensions (80% of original size)
ICON_SIZE = 9.6
ICON_CENTER = 4.8
WEATHER_CELL_WIDTH = 19.2

# Table column widths in mm
COLUMN_WIDTHS = [10, 10, 10, 10, 11, 32, 10, 10, 11, 10, 16, 10, 10, 20, 10, 32]

# GeoNames API configuration
def get_api_key():
    """
    Get API key from environment variable (GitHub Actions) or local secrets file
    """
    # First, try to get from environment variable (GitHub Actions)
    api_key = os.getenv('GEONAMES_API_KEY')

    if api_key:
        return api_key

    # If not found in environment, try to load from local secrets file
    secrets_file = os.path.join('secrets', 'geonames.json')

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

GEONAMES_USERNAME = get_api_key()

# ============================================================================
# NAVIGATION CALCULATIONS
# ============================================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula.

    Returns:
        Distance in nautical miles
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_NM * c


def format_position(lat, lon):
    """Format coordinates in degrees and decimal minutes (DD° MM.mmm' N/S).

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        Formatted string like "47° 33.456' N 8° 12.345' E"
    """
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_dir = 'N' if lat >= 0 else 'S'

    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_dir = 'E' if lon >= 0 else 'W'

    return f"{lat_deg}° {lat_min:.3f}' {lat_dir} {lon_deg}° {lon_min:.3f}' {lon_dir}"


def get_wind_direction_abbr(degrees):
    """Convert wind direction degrees to compass abbreviation (N, NE, etc)."""
    if pd.isna(degrees):
        return "---"
    index = int((degrees + 11.25) / 22.5) % 16
    return WIND_DIRECTIONS[index]

def get_sea_state_description(wave_height):
    """
    Convert wave height to German sea state description (Seegang).

    Args:
        wave_height: Wave height in meters

    Returns:
        english sea state description
    """
    if pd.isna(wave_height):
        return "---"

    if wave_height == 0:
        return "calm-glassy"
    elif wave_height <= 0.1:
        return "calm-rippled"
    elif wave_height <= 0.5:
        return "smooth"
    elif wave_height <= 1.25:
        return "slight"
    elif wave_height <= 2.5:
        return "moderate"
    elif wave_height <= 4.0:
        return "rough"
    elif wave_height <= 6.0:
        return "very rough"
    elif wave_height <= 9.0:
        return "high"
    elif wave_height <= 14.0:
        return "very high"
    else:
        return "pheonomenal"

def get_bearing_text(lat1, lon1, lat2, lon2):
    """Calculate bearing and return direction text (N, NE, SW, etc)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    bearing = (bearing + 360) % 360

    # Convert to direction text
    index = int((bearing + 11.25) / 22.5) % 16
    return WIND_DIRECTIONS[index]


def get_nearest_location(lat, lon, username=GEONAMES_USERNAME):
    """
    Get nearest location using GeoNames API.
    First tries to find major cities within 10 nm, then falls back to nearest place.

    Args:
        lat: Latitude
        lon: Longitude
        username: GeoNames username (register at geonames.org)

    Returns:
        Dictionary with location info or None if request fails
    """
    try:
        # First, search for major cities within 10 nm (~18.5 km)
        url = "http://api.geonames.org/findNearbyPlaceNameJSON"
        params = {
            'lat': lat,
            'lng': lon,
            'username': username,
            'radius': 20,  # 20 km to cover ~10 nm
            'maxRows': 10,  # Get multiple results
            'style': 'FULL',
            'cities': 'cities15000'  # Only cities with population > 15,000
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if 'geonames' in data and len(data['geonames']) > 0:
                # Sort by population (descending) to get the largest city
                places = data['geonames']
                places_with_pop = [p for p in places if p.get('population', 0) > 0]

                if places_with_pop:
                    # Get the most populous city
                    largest_city = max(places_with_pop, key=lambda p: p.get('population', 0))

                    return {
                        'name': largest_city.get('name', 'Unknown'),
                        'country': largest_city.get('countryCode', ''),
                        'lat': float(largest_city.get('lat', lat)),
                        'lon': float(largest_city.get('lng', lon)),
                        'feature': largest_city.get('fclName', ''),
                        'distance': float(largest_city.get('distance', 0)),
                        'adminName1': largest_city.get('adminName1', ''),
                        'population': largest_city.get('population', 0)
                    }

        # Fallback: Search for any nearby place (wider radius)
        params = {
            'lat': lat,
            'lng': lon,
            'username': username,
            'radius': 300,  # Wider search
            'maxRows': 1,
            'style': 'FULL'
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200 and response.text != '{"geonames":[]}':
            data = response.json()
            if 'geonames' in data and len(data['geonames']) > 0:
                place = data['geonames'][0]
                return {
                    'name': place.get('name', 'Unknown'),
                    'country': place.get('countryCode', ''),
                    'lat': float(place.get('lat', lat)),
                    'lon': float(place.get('lng', lon)),
                    'feature': place.get('fclName', ''),
                    'distance': float(place.get('distance', 0)),
                    'adminName1': place.get('adminName1', ''),
                }

        # If still no results, try ocean names
        if response.status_code == 200 and response.text == '{"geonames":[]}':
            url = "http://api.geonames.org/oceanJSON"
            params = {
                'lat': lat,
                'lng': lon,
                'username': username,
                'style': 'FULL'
            }

            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            if 'ocean' in data and len(data['ocean']) > 0:
                place = data['ocean']
                return {
                    'name': place.get('name', 'Unknown'),
                    'country': '',
                    'lat': float(lat),
                    'lon': float(lon),
                    'feature': '',
                    'distance': float(place.get('distance', 0)),
                    'adminName1': '',
                }

    except Exception as e:
        print(f"GeoNames API error: {e}")

    return None


def format_location_with_distance(lat, lon, username=GEONAMES_USERNAME):
    """
    Format location as "15 nm SW of Brest, FR"

    Args:
        lat: Latitude
        lon: Longitude
        username: GeoNames username

    Returns:
        Formatted string with distance and direction to nearest place
    """
    location = get_nearest_location(lat, lon, username)

    if location:
        # Calculate distance in nautical miles
        distance = calculate_distance(lat, lon, location['lat'], location['lon'])

        # Get bearing direction
        bearing = get_bearing_text(location['lat'], location['lon'], lat, lon)

        #Define the core parts of the location
        location_parts = [
        location.get('name'),
        location.get('adminName1'),
        location.get('country')]

        # filtering out any empty or None values.
        formatted_location = ", ".join([part for part in location_parts if part])

        # Format output
        if distance < 0.5:
            return formatted_location
        else:
            return f"{distance:.1f} nm {bearing} von {formatted_location}"
    else:
        # Fallback to coordinates only
        return "-"


# ============================================================================
# WEATHER SYMBOLS
# ============================================================================

def create_wind_barb(wind_speed, wind_dir):
    """Create meteorological wind barb symbol.

    Wind barbs show direction (where wind blows TO) and speed:
    - Half barb = 5 knots
    - Full barb = 10 knots
    """
    d = Drawing(ICON_SIZE, ICON_SIZE)

    if pd.isna(wind_speed) or pd.isna(wind_dir) or wind_speed == 0:
        return d

    # Convert to radians (add 180 to show where wind goes TO)
    rad = math.radians(wind_dir + 180)

    # Draw staff line
    staff_len = 6.0
    x1, y1 = ICON_CENTER, ICON_CENTER
    x2 = x1 + staff_len * math.sin(rad)
    y2 = y1 + staff_len * math.cos(rad)
    d.add(Line(x1, y1, x2, y2, strokeWidth=0.72))

    # Add barbs based on speed
    full_barbs = int(wind_speed / 10)
    half_barbs = 1 if (wind_speed % 10) >= 5 else 0

    barb_spacing = 1.52
    current_pos = 0

    # Draw full barbs (10 knots each)
    for _ in range(full_barbs):
        bx = x2 - current_pos * math.sin(rad)
        by = y2 - current_pos * math.cos(rad)

        perp_rad = rad + math.radians(90)
        barb_len = 2.08
        bx2 = bx + barb_len * math.sin(perp_rad)
        by2 = by + barb_len * math.cos(perp_rad)

        d.add(Line(bx, by, bx2, by2, strokeWidth=0.72))
        current_pos += barb_spacing

    # Draw half barb (5 knots)
    if half_barbs:
        bx = x2 - current_pos * math.sin(rad)
        by = y2 - current_pos * math.cos(rad)

        perp_rad = rad + math.radians(90)
        barb_len = 1.04
        bx2 = bx + barb_len * math.sin(perp_rad)
        by2 = by + barb_len * math.cos(perp_rad)

        d.add(Line(bx, by, bx2, by2, strokeWidth=0.72))

    return d


def create_cloud_cover(cloud_cover):
    """Create cloud cover circle symbol.

    Circle is filled based on percentage:
    - 0-25%: Empty circle (clear)
    - 25-50%: Quarter filled
    - 50-75%: Half filled
    - 75-100%: Fully filled (overcast)
    """
    d = Drawing(ICON_SIZE, ICON_SIZE)

    circle = Circle(ICON_CENTER, ICON_CENTER, 3.6)
    circle.strokeWidth = 0.72
    circle.strokeColor = colors.black

    if pd.isna(cloud_cover):
        circle.fillColor = colors.white
        d.add(circle)
        return d

    if cloud_cover < 25:
        circle.fillColor = colors.white
        d.add(circle)
    elif cloud_cover < 50:
        circle.fillColor = colors.white
        d.add(circle)
        wedge = Wedge(ICON_CENTER, ICON_CENTER, 3.6, 90, 180,
                     fillColor=colors.black, strokeColor=None)
        d.add(wedge)
    elif cloud_cover < 75:
        circle.fillColor = colors.white
        d.add(circle)
        wedge = Wedge(ICON_CENTER, ICON_CENTER, 3.6, 0, 180,
                     fillColor=colors.black, strokeColor=None)
        d.add(wedge)
    else:
        circle.fillColor = colors.black
        d.add(circle)

    return d


def create_weather_cell(wind_speed, wind_dir, cloud_cover):
    """Create combined weather symbol with wind barb and cloud cover side by side."""
    d = Drawing(WEATHER_CELL_WIDTH, ICON_SIZE)

    # Add wind barb on left
    wind_barb = create_wind_barb(wind_speed, wind_dir)
    for item in wind_barb.contents:
        d.add(item)

    # Add cloud cover on right (shifted)
    cloud = create_cloud_cover(cloud_cover)
    for item in cloud.contents:
        if hasattr(item, 'x'):
            item.x += ICON_SIZE
        elif hasattr(item, 'x1'):
            item.x1 += ICON_SIZE
            item.x2 += ICON_SIZE
        if hasattr(item, 'cx'):
            item.cx += ICON_SIZE
        if isinstance(item, Wedge):
            item.centerx += ICON_SIZE
        d.add(item)

    return d


# ============================================================================
# CHARTS AND MAPS
# ============================================================================

def create_pressure_chart(day_data):
    """Create atmospheric pressure chart for the day."""
    fig, ax = plt.subplots(figsize=(2.8, 2.0))

    times_str = day_data['time'].values
    pressures = day_data['luftdruck'].dropna().values

    if len(pressures) > 0:
        # Convert times to hours (0-24)
        time_objs = [datetime.strptime(t, '%H:%M') for t in times_str[:len(pressures)]]
        time_hours = [t.hour + t.minute/60 for t in time_objs]

        # Plot pressure curve
        ax.plot(time_hours, pressures, 'k-', linewidth=2, marker='o', markersize=3)

        # Configure Y-axis (pressure)
        y_min, y_max = pressures.min(), pressures.max()
        y_ticks = np.linspace(y_min, y_max, 8)
        ax.set_ylim(y_min - 0.2, y_max + 0.2)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=7)
        ax.set_ylabel('mbar', fontsize=8)

        # Configure X-axis (time 0-24 hours)
        ax.set_xlim(0, 24)
        ax.set_xticks([6, 12, 18])
        ax.set_xticklabels(['06:00', '12:00', '18:00'], fontsize=8)

        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def create_track_map(day_data, single_map=False):
    """Create track map with satellite basemap (if available)."""
    lats = day_data['latitude'].values
    lons = day_data['longitude'].values
    speeds = day_data['sog'].values


    # Handle single point
    if single_map:
        return _create_single_point_map(lats[0], lons[0])

    # Create map with track line
    return _create_geomap_with_track(lats, lons, speeds)



def _lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def _get_tile_bbox(xtile, ytile, zoom):
    """Get bounding box of a tile in EPSG:3857."""
    from pyproj import Transformer

    n = 2.0 ** zoom
    lon_min = xtile / n * 360.0 - 180.0
    lon_max = (xtile + 1) / n * 360.0 - 180.0

    lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
    lat_min = math.degrees(lat_rad_min)
    lat_max = math.degrees(lat_rad_max)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lon_min, lat_min)
    xmax, ymax = transformer.transform(lon_max, lat_max)

    return xmin, ymin, xmax, ymax

def _fetch_and_composite_tiles(xmin, ymin, xmax, ymax, zoom):
    """Fetch Esri tiles for the bounding box and composite them."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    # Get tile range
    xtile_min, ytile_max = _lat_lon_to_tile(lat_min, lon_min, zoom)
    xtile_max, ytile_min = _lat_lon_to_tile(lat_max, lon_max, zoom)

    imagery_tiles = []
    label_tiles = []

    for ytile in range(ytile_min, ytile_max + 1):
        row_imagery = []
        row_labels = []
        for xtile in range(xtile_min, xtile_max + 1):
            # Esri World Imagery
            imagery_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}'
            # Esri Labels
            label_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{zoom}/{ytile}/{xtile}'

            try:
                img_response = requests.get(imagery_url, timeout=5)
                img = Image.open(BytesIO(img_response.content))
                row_imagery.append(np.array(img))

                label_response = requests.get(label_url, timeout=5)
                label_img = Image.open(BytesIO(label_response.content))
                row_labels.append(np.array(label_img))
            except Exception as e:
                print(f"Failed to fetch tile {xtile},{ytile}: {e}")
                row_imagery.append(np.zeros((256, 256, 3), dtype=np.uint8))
                row_labels.append(np.zeros((256, 256, 4), dtype=np.uint8))

        if row_imagery:
            imagery_tiles.append(np.concatenate(row_imagery, axis=1))
            label_tiles.append(np.concatenate(row_labels, axis=1))

    if imagery_tiles:
        composite_imagery = np.concatenate(imagery_tiles, axis=0)
        composite_labels = np.concatenate(label_tiles, axis=0)

        # Get extent of the composite image
        tile_xmin, tile_ymin, _, _ = _get_tile_bbox(xtile_min, ytile_max, zoom)
        _, _, tile_xmax, tile_ymax = _get_tile_bbox(xtile_max, ytile_min, zoom)

        return composite_imagery, composite_labels, (tile_xmin, tile_xmax, tile_ymin, tile_ymax)

    return None, None, None

def _calculate_zoom_level(dimension_meters):
    """Calculate optimal zoom level based on dimension in meters.
    Maximum zoom level is 14 to avoid over-zooming."""
    return max(1, min(13, round(20 - math.log2(dimension_meters / 100))))

def _add_basemap_to_axis(ax, bounds, zoom):
    """Add Esri satellite imagery and labels to the axis.

    Args:
        ax: Matplotlib axis
        bounds: Tuple of (xmin, xmax, ymin, ymax) in EPSG:3857
        zoom: Zoom level for tiles
    """
    try:
        imagery, labels, extent = _fetch_and_composite_tiles(
            bounds[0], bounds[2], bounds[1], bounds[3], zoom)

        if imagery is not None:
            ax.imshow(imagery, extent=extent, zorder=1, interpolation='bilinear')
            ax.imshow(labels, extent=extent, zorder=2, interpolation='bilinear', alpha=1.0)
    except Exception as e:
        print(f"Could not load map tiles: {e}")
        ax.set_facecolor('#E8F4F8')

def _add_scalebar(ax):
    """Add a scalebar to the axis."""
    from matplotlib_scalebar.scalebar import ScaleBar

    scalebar = ScaleBar(
        1,  # 1 unit = 1 meter
        "m",
        location="lower left",
        frameon=False,
        color="white",
        font_properties={"size": 10}
    )
    ax.add_artist(scalebar)

def _create_geomap_with_track(lats, lons, speeds):
    """Create map with vessel track colored by speed."""
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from io import BytesIO

    # Create geometries
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    line = LineString([(lon, lat) for lon, lat in zip(lons, lats)])

    # Create GeoDataFrames and convert to Web Mercator
    gdf_line = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:4326')
    gdf_points = gpd.GeoDataFrame({
        'type': ['start', 'end'],
        'geometry': [points[0], points[-1]]
    }, crs='EPSG:4326')

    gdf_line_3857 = gdf_line.to_crs('EPSG:3857')
    gdf_points_3857 = gdf_points.to_crs('EPSG:3857')

    # Calculate bounds with proper aspect ratio
    xmin, ymin, xmax, ymax = gdf_line_3857.total_bounds
    x_range = max(xmax - xmin, 100)
    y_range = max(ymax - ymin, 100)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # Add padding and adjust for figure aspect ratio
    x_padded = x_range * 1.1
    y_padded = y_range * 1.1

    fig_aspect = 3.5 / 2.0
    data_aspect = x_padded / y_padded

    if data_aspect > fig_aspect:
        y_padded = x_padded / fig_aspect
    else:
        x_padded = y_padded * fig_aspect

    # Calculate zoom level
    max_dim_m = max(x_padded, y_padded)
    zoom = _calculate_zoom_level(max_dim_m)
    print(f"Track map - Using zoom level: {zoom} for dimension: {max_dim_m:.0f}m")

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    bounds = (x_center - x_padded/2, x_center + x_padded/2,
              y_center - y_padded/2, y_center + y_padded/2)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal', adjustable='datalim')

    # Add basemap
    _add_basemap_to_axis(ax, bounds, zoom)

    # Plot track segments with speed-based coloring
    max_speed = max(speeds.max(), 1)
    for i in range(len(lons)-1):
        segment = LineString([(lons[i], lats[i]), (lons[i+1], lats[i+1])])
        gdf_seg = gpd.GeoDataFrame({'geometry': [segment]}, crs='EPSG:4326')
        gdf_seg_3857 = gdf_seg.to_crs('EPSG:3857')

        # Color based on speed
        speed_ratio = speeds[i] / max_speed
        if speed_ratio < 0.5:
            color = plt.cm.YlOrRd(speed_ratio * 2 * 0.3)
        else:
            color = plt.cm.YlOrRd(0.3 + (speed_ratio - 0.5) * 2 * 0.7)

        gdf_seg_3857.plot(ax=ax, color=color, linewidth=3, alpha=0.9, zorder=10)

    # Add scalebar
    _add_scalebar(ax)

    # Add start/end markers
    start = gdf_points_3857[gdf_points_3857['type'] == 'start']
    end = gdf_points_3857[gdf_points_3857['type'] == 'end']

    start.plot(ax=ax, color='#00AA00', markersize=60,
              edgecolor='white', linewidth=1.5, zorder=15, marker='o')
    end.plot(ax=ax, color='#CC0000', markersize=80,
            edgecolor='white', linewidth=1.5, zorder=15, marker='^')

    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf

def _create_single_point_map(lat, lon):
    """Create map for a single position (anchored vessel)."""
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point
    from io import BytesIO

    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
    gdf_3857 = gdf.to_crs('EPSG:3857')

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Get point coordinates
    x, y = gdf_3857.geometry.iloc[0].x, gdf_3857.geometry.iloc[0].y

    # Set initial padding
    base_padding = 2500
    x_padded = base_padding * 2
    y_padded = base_padding * 2

    # Adjust for figure aspect ratio (matching track map logic)
    fig_aspect = 3.5 / 2.0
    data_aspect = x_padded / y_padded

    if data_aspect > fig_aspect:
        y_padded = x_padded / fig_aspect
    else:
        x_padded = y_padded * fig_aspect

    # Calculate bounds centered on point
    bounds = (x - x_padded/2, x + x_padded/2,
              y - y_padded/2, y + y_padded/2)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal', adjustable='datalim')

    # Calculate zoom level
    max_dim_m = max(x_padded, y_padded)
    zoom = _calculate_zoom_level(max_dim_m)
    print(f"Single point map - Using zoom level: {zoom} for dimension: {max_dim_m:.0f}m")

    # Add basemap
    _add_basemap_to_axis(ax, bounds, zoom)

    # Plot point
    gdf_3857.plot(ax=ax, color='#00AA00', markersize=100,
                  edgecolor='white', linewidth=2, zorder=15, marker='o')

    # Add scalebar
    _add_scalebar(ax)

    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf



# ============================================================================
# DATA PROCESSING
# ============================================================================

def sample_hourly_data(day_data, max_entries=19):
    """Sample data to fit on one page (max 19 entries + 3 summary rows).
    For anchored days (total distance = 0), show only first and last entry."""

    # Calculate total distance for the day to check if anchored
    total_distance = 0
    for i in range(1, len(day_data)):
        dist = calculate_distance(
            day_data.iloc[i-1]['latitude'], day_data.iloc[i-1]['longitude'],
            day_data.iloc[i]['latitude'], day_data.iloc[i]['longitude']
        )
        total_distance += dist

    # If anchored (distance < 0.1 nm), show only first and last
    if total_distance < 0.1 and len(day_data) > 1:
        return pd.concat([
            day_data.iloc[0:1].copy(),
            day_data.iloc[-1:].copy()
        ]).reset_index(drop=True)

    # Original sampling logic for non-anchored days
    if len(day_data) <= max_entries:
        return day_data.reset_index(drop=True)

    # Keep first and last entries
    first = day_data.iloc[0:1].copy()
    last = day_data.iloc[-1:].copy()
    middle = day_data.iloc[1:-1].copy()

    # Sample middle by hour
    middle.loc[:, 'hour'] = pd.to_datetime(middle['timestamp_utc']).dt.hour
    sampled_middle = middle.groupby('hour').first().reset_index(drop=True)

    # Combine and sort
    result = pd.concat([first, sampled_middle, last])
    result = result.drop_duplicates(subset=['timestamp_utc'])
    result = result.sort_values('timestamp_utc').reset_index(drop=True)

    # Further limit if needed
    if len(result) > max_entries:
        indices = [0] + list(np.linspace(1, len(result)-2, max_entries-2, dtype=int)) + [len(result)-1]
        result = result.iloc[indices].copy()

    return result.reset_index(drop=True)


def format_table_row(row, day_data, total_log, anchor_img, sailing_img):
    """Format a single logbook table row."""
    # Get status icon
    nav_status = row.get('navigational_status', 0)
    sog = row.get('sog', 0)
    if pd.isna(sog) or sog == 999:
        sog = 0

    status_icon = None
    if (nav_status == 1 or sog == 0) and anchor_img:
        status_icon = RLImage(anchor_img, width=3.2*mm, height=3.2*mm)
    elif sog > 0 and sailing_img:
        status_icon = RLImage(sailing_img, width=3.2*mm, height=3.2*mm)

    # Wind data
    wind_dir_abbr = get_wind_direction_abbr(row.get('windrichtung'))
    wind_kn = f"{row['windstaerke']:.1f}" if pd.notna(row.get('windstaerke')) else "---"

    # Weather cell
    weather_cell = create_weather_cell(
        row.get('windstaerke'),
        row.get('windrichtung'),
        row.get('bewoelkung')
    )

    # Weather condition text
    witterung = row.get('wetterzustand', '')
    if pd.notna(witterung) and len(str(witterung)) > 24:
        witterung = str(witterung)[:24]
    elif pd.isna(witterung):
        witterung = ''

    # Get sea state description
    sea_state = get_sea_state_description(row.get('wellenhoehe'))

    # Format numeric values
    def fmt(val, fmt_str="---"):
        return f"{val:{fmt_str}}" if pd.notna(val) else "---"

    cog = row.get('cog')
    if pd.isna(cog) or cog == 999 or cog == 360:
        cog_str = "---"
    else:
        cog_str = f"{cog:.0f}°"

    sog_str = "0.0" if sog == 0 else f"{sog:.1f}"

    return [
        row['time'],
        status_icon if status_icon else "",
        wind_dir_abbr,
        wind_kn,
        weather_cell,
        witterung,
        fmt(row.get('luftdruck'), '.1f'),
        fmt(row.get('lufttemperatur'), '.1f'),
        fmt(row.get('wassertemperatur'), '.1f'),
        fmt(row.get('wellenhoehe'), '.2f'),
        sea_state,  # NEW COLUMN
        fmt(row.get('niederschlag'), '.1f'),
        cog_str,
        sog_str,
        f"{total_log:.1f}",
        format_position(row['latitude'], row['longitude'])[:25]  # Reduced from 30 to 25
    ]


# ============================================================================
# PDF GENERATION
# ============================================================================

def create_title_page(styles, df):
    """Create professional title page with voyage information (fits A4 landscape).

    Args:
        styles: ReportLab styles
        df: DataFrame with voyage data to extract dates and vessel info
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.platypus import Table, TableStyle

    # Custom styles - REDUCED sizes for landscape
    title_bold = ParagraphStyle(
        'CustomTitle',
        parent=styles['Normal'],
        fontSize=36,
        leading=42,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        textColor=HexColor('#003366')
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=24,
        leading=28,
        fontName='Helvetica',
        alignment=TA_CENTER,
        textColor=HexColor('#003366')
    )

    ship_style = ParagraphStyle(
        'CustomShip',
        parent=styles['Normal'],
        fontSize=20,
        leading=24,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    info_label_style = ParagraphStyle(
        'InfoLabel',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        alignment=TA_RIGHT
    )

    info_value_style = ParagraphStyle(
        'InfoValue',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        alignment=TA_LEFT
    )

    # Extract voyage information from data
    start_date = df['timestamp_utc'].min().strftime('%d %B %Y')
    end_date = df['timestamp_utc'].max().strftime('%d %B %Y')

    # Get MMSI from data, fallback to config
    mmsi = df['mmsi'].iloc[0] if 'mmsi' in df.columns else VESSEL_INFO.get('mmsi', 'N/A')

    # Calculate total voyage distance
    total_distance = 0
    for i in range(1, len(df)):
        dist = calculate_distance(
            df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
            df.iloc[i]['latitude'], df.iloc[i]['longitude']
        )
        total_distance += dist

    # Get start and end positions
    start_lat = df.iloc[0]['latitude']
    start_lon = df.iloc[0]['longitude']
    end_lat = df.iloc[-1]['latitude']
    end_lon = df.iloc[-1]['longitude']

    start_location = format_location_with_distance(start_lat, start_lon)
    end_location = format_location_with_distance(end_lat, end_lon)

    # Voyage duration
    duration = (df['timestamp_utc'].max() - df['timestamp_utc'].min()).days + 1

    elements = [
        Spacer(1, 30*mm),
        Paragraph(LOGBOOK_SETTINGS.get('title', 'NAUTICAL LOGBOOK'), title_bold),
        Spacer(1, 5*mm),
        Paragraph(LOGBOOK_SETTINGS.get('subtitle', 'Voyage Log'), subtitle_style),
        Spacer(1, 8*mm),
        Paragraph(f"{VESSEL_INFO.get('prefix', 'M/V')} {VESSEL_INFO.get('name', 'Unknown')}", ship_style),
        Spacer(1, 12*mm),
    ]

    # Build voyage information table dynamically based on settings
    voyage_info = []

    # Always show MMSI if enabled
    if LOGBOOK_SETTINGS.get('show_mmsi', True):
        voyage_info.append([
            Paragraph("MMSI:", info_label_style),
            Paragraph(f"{mmsi}", info_value_style)
        ])

    # Optional: IMO Number
    if LOGBOOK_SETTINGS.get('show_imo', False) and VESSEL_INFO.get('imo'):
        voyage_info.append([
            Paragraph("IMO Number:", info_label_style),
            Paragraph(VESSEL_INFO['imo'], info_value_style)
        ])

    # Optional: Call Sign
    if LOGBOOK_SETTINGS.get('show_call_sign', False) and VESSEL_INFO.get('call_sign'):
        voyage_info.append([
            Paragraph("Call Sign:", info_label_style),
            Paragraph(VESSEL_INFO['call_sign'], info_value_style)
        ])

    # Optional: Flag State
    if LOGBOOK_SETTINGS.get('show_flag_state', True) and VESSEL_INFO.get('flag_state'):
        voyage_info.append([
            Paragraph("Flaggenstaat:", info_label_style),
            Paragraph(VESSEL_INFO['flag_state'], info_value_style)
        ])

    # Optional: Vessel Type
    if LOGBOOK_SETTINGS.get('show_vessel_type', True) and VESSEL_INFO.get('vessel_type'):
        voyage_info.append([
            Paragraph("Schiffstyp:", info_label_style),
            Paragraph(VESSEL_INFO['vessel_type'], info_value_style)
        ])

    # Optional: Dimensions
    if LOGBOOK_SETTINGS.get('show_dimensions', False):
        if VESSEL_INFO.get('length') and VESSEL_INFO.get('beam'):
            voyage_info.append([
                Paragraph("Abmessungen:", info_label_style),
                Paragraph(f"L: {VESSEL_INFO['length']}m × B: {VESSEL_INFO['beam']}m × T: {VESSEL_INFO.get('draft', 'N/A')}m",
                         info_value_style)
            ])

    # Optional: Master
    if LOGBOOK_SETTINGS.get('show_master', False) and VESSEL_INFO.get('master'):
        voyage_info.append([
            Paragraph("Kapitän:", info_label_style),
            Paragraph(VESSEL_INFO['master'], info_value_style)
        ])

    # Always show voyage period
    voyage_info.append([
        Paragraph("Reisezeitraum:", info_label_style),
        Paragraph(f"{start_date} — {end_date}", info_value_style)
    ])

    # Always show duration
    voyage_info.append([
        Paragraph("Dauer:", info_label_style),
        Paragraph(f"{duration} Tage", info_value_style)
    ])

    # Always show total distance
    voyage_info.append([
        Paragraph("Gesamtdistanz:", info_label_style),
        Paragraph(f"{total_distance:.1f} Seemeilen", info_value_style)
    ])

    # Always show departure
    voyage_info.append([
        Paragraph("Abfahrt:", info_label_style),
        Paragraph(f"{start_location}", info_value_style)
    ])

    # Always show arrival
    voyage_info.append([
        Paragraph("Ankunft:", info_label_style),
        Paragraph(f"{end_location}", info_value_style)
    ])

    info_table = Table(voyage_info, colWidths=[50*mm, 110*mm])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.grey),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
    ]))

    elements.append(info_table)
    elements.append(Spacer(1, 10*mm))

    # Add footer note
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica-Oblique',
        alignment=TA_CENTER,
        textColor=colors.grey
    )

    elements.append(Paragraph(
        "Dieses Logbuch wurde automatisch aus AIS-Positionsmeldungen (https://aisstream.io/) und Wetterdaten (https://open-meteo.com/) generiert. Durch die Nutzung dieser Datenquellen können Abweichungen und Fehler auftreten. Insbesondere AIS-Daten können in Folge fehlender Datenübertragungen unvollständig oder fehlerhaft sein.",
        footer_style
    ))

    elements.append(PageBreak())

    return elements

def create_ship_track_page(geojson_path):
    """Create a full-page map showing the ship's historical track from GeoJSON."""
    import json
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from io import BytesIO
    from reportlab.platypus import Image, PageBreak
    from reportlab.lib.utils import ImageReader

    # Read GeoJSON - it already contains LineString geometry
    gdf = gpd.read_file(geojson_path)

    if len(gdf) == 0:
        print("No features found in GeoJSON")
        return [PageBreak()]

    # Convert to Web Mercator
    gdf_3857 = gdf.to_crs('EPSG:3857')

    # Calculate bounds
    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    x_range = max(xmax - xmin, 100)
    y_range = max(ymax - ymin, 100)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # Add padding
    x_padded = x_range * 1.1
    y_padded = y_range * 1.1

    # Landscape aspect ratio
    page_aspect = 297.0 / 210.0
    data_aspect = x_padded / y_padded

    if data_aspect > page_aspect:
        y_padded = x_padded / page_aspect
    else:
        x_padded = y_padded * page_aspect

    # Calculate zoom level
    max_dim_m = max(x_padded, y_padded)
    zoom = _calculate_zoom_level(max_dim_m)
    print(f"Ship track page - Using zoom level: {zoom} for dimension: {max_dim_m:.0f}m")

    # Create figure in landscape orientation
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    bounds = (x_center - x_padded/2, x_center + x_padded/2,
              y_center - y_padded/2, y_center + y_padded/2)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # Add basemap
    _add_basemap_to_axis(ax, bounds, zoom)

    # Plot track as red dashed line
    gdf_3857.plot(
        ax=ax,
        color='red',
        linewidth=2.5,
        linestyle='--',
        alpha=0.8,
        zorder=10
    )

    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    # Calculate exact dimensions that will fit in the frame
    # Frame size: 773.197 x 498.236 points
    max_width = 773.197
    max_height = 498.236

    # Get image dimensions
    img_reader = ImageReader(buf)
    img_width, img_height = img_reader.getSize()

    # Calculate scaling to fit within frame while maintaining aspect ratio
    width_scale = max_width / img_width
    height_scale = max_height / img_height
    scale = min(width_scale, height_scale) * 0.95  # 95% to ensure it fits

    final_width = img_width * scale
    final_height = img_height * scale

    print(f"Ship track image: {final_width:.2f} x {final_height:.2f} points")

    # Reset buffer position
    buf.seek(0)

    # Create image with exact calculated dimensions
    img = Image(buf, width=final_width, height=final_height)

    return [img, PageBreak()]

def get_image_with_aspect(img_buffer, max_width_mm, max_height_mm):
    """Create ReportLab Image with preserved aspect ratio."""
    from reportlab.lib import utils

    img = utils.ImageReader(img_buffer)
    iw, ih = img.getSize()
    aspect = ih / float(iw)

    width_mm = max_width_mm
    height_mm = width_mm * aspect

    if height_mm > max_height_mm:
        height_mm = max_height_mm
        width_mm = height_mm / aspect

    return RLImage(img_buffer, width=width_mm*mm, height=height_mm*mm)


def create_day_page(date, day_data, cumulative_log, previous_day_log,
                    styles, anchor_img, sailing_img,
                    last_pos_prev_day={'lat': None, 'lon': None}):
    """Create a complete day page with header, charts, and logbook table."""
    elements = []

    # Date header
    locale.setlocale(locale.LC_TIME, 'de_CH.UTF-8')
    date_str = date.strftime('%A, %d. %B %Y')
    header = Paragraph(f"{date_str}", styles['Heading2'])
    elements.append(header)
    elements.append(Spacer(1, 2*mm))


    # Calculate distance from previous day's last position to current day's first position
    day_log = 0
    if last_pos_prev_day['lat'] is not None and len(day_data) > 0:
        dist_offset = calculate_distance(
            last_pos_prev_day['lat'], last_pos_prev_day['lon'],
            day_data.iloc[0]['latitude'], day_data.iloc[0]['longitude']
        )
        day_log += dist_offset
    for idx, row in day_data.iterrows():
        row_position = day_data.index.get_loc(idx)

        if row_position > 0:
            prev_idx = day_data.index[row_position - 1]
            prev = day_data.loc[prev_idx]
            dist = calculate_distance(prev['latitude'], prev['longitude'],
                                    row['latitude'], row['longitude'])
            day_log += dist

    # Determine if single point map is needed (single measurement) or anchored (no movement)
    if len(day_data) < 2 or day_log <= 0.09:
        single_map = True
    else:
        single_map = False
    
    # Create charts
    pressure_chart = create_pressure_chart(day_data)
    track_map = create_track_map(day_data, single_map=single_map)

    pressure_img = get_image_with_aspect(pressure_chart, 60, 43)
    track_img = get_image_with_aspect(track_map, 93, 40)

    # Position info
    first_pos = format_position(day_data.iloc[0]['latitude'],
                               day_data.iloc[0]['longitude'])
    last_pos = format_position(day_data.iloc[-1]['latitude'],
                              day_data.iloc[-1]['longitude'])

    # Get nearest locations with distance
    first_location = format_location_with_distance(
        day_data.iloc[0]['latitude'],
        day_data.iloc[0]['longitude']
    )
    last_location = format_location_with_distance(
        day_data.iloc[-1]['latitude'],
        day_data.iloc[-1]['longitude']
    )

    from_to_text = (f"<b>VON</b> {first_pos}<br/>"
                   f"{first_location}<br/><br/>"
                   f"<b>NACH</b> {last_pos}<br/>"
                   f"{last_location}")
    from_to = Paragraph(from_to_text, styles['Normal'])

    # Create header table with charts
    data_table_width = sum(COLUMN_WIDTHS)

    header_table = Table(
        [[from_to, pressure_img, track_img]],
        colWidths=[(data_table_width - 153)*mm, 60*mm, 93*mm]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 3*mm))

    # Create logbook table with NEW HEADER
    table_data = [[
        'Zeit\nUTC', 'Status', 'Wind\nR.', 'Wind\nkn', 'Wetter', 'Witterung',
        'L.Druck\nmbar', 'T.Luft\n°C', 'T.Wasser\n°C', 'Wellen\nm', 'Seegang\n Douglas',
        'Regen\nmm', 'Kurs\n°', 'Fahrt\nkn', 'Log\nsm', 'Position'
    ]]

    # Add data rows

    for idx, row in day_data.iterrows():
        row_position = day_data.index.get_loc(idx)

        if row_position > 0:
            prev_idx = day_data.index[row_position - 1]
            prev = day_data.loc[prev_idx]
            dist = calculate_distance(prev['latitude'], prev['longitude'],
                                    row['latitude'], row['longitude'])
            day_log += dist

        table_data.append(format_table_row(row, day_data, day_log,
                                          anchor_img, sailing_img))

    # Add summary rows
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      '', 'Tagessumme', f'{day_log:.1f}', ''])
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      '', 'Vortrag', f'{previous_day_log:.1f}', ''])

    cumulative_log += day_log
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      '', 'Gesamt', f'{cumulative_log:.1f}', ''])

    # Create and style table
    col_widths_mm = [w*mm for w in COLUMN_WIDTHS]
    table = Table(table_data, colWidths=col_widths_mm, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTSIZE', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 0.5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 0.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, -3), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, -3), (-1, -1), 'Helvetica-Bold'),
    ]))

    elements.append(table)
    elements.append(PageBreak())

    return elements, cumulative_log

def extract_date_range_from_pdf(pdf_path):
    """
    Extract start and end dates from the title page of existing PDF.

    Returns:
        tuple: (start_date, end_date) as datetime objects, or (None, None) if not found
    """
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) == 0:
            return None, None

        # Extract text from first page
        first_page = reader.pages[0]
        text = first_page.extract_text()

        # Look for date pattern: "DD Month YYYY — DD Month YYYY"
        # Example: "17 October 2025 — 20 November 2025"
        date_pattern = r'(\d{1,2}\s+\w+\s+\d{4})\s*—\s*(\d{1,2}\s+\w+\s+\d{4})'
        match = re.search(date_pattern, text)

        if match:
            start_str = match.group(1)
            end_str = match.group(2)

            # Parse dates
            start_date = datetime.strptime(start_str, '%d %B %Y')
            end_date = datetime.strptime(end_str, '%d %B %Y')

            return start_date, end_date

    except Exception as e:
        print(f"Error extracting dates from PDF: {e}")

    return None, None


def needs_update(pdf_path, csv_file):
    """
    Check if PDF needs updating based on date range.

    Returns:
        tuple: (needs_update: bool, missing_start_date: datetime or None, missing_end_date: datetime or None)
    """
    if not os.path.exists(pdf_path):
        return True, None, None  # PDF doesn't exist, needs full generation

    # Extract dates from existing PDF
    pdf_start, pdf_end = extract_date_range_from_pdf(pdf_path)

    if pdf_start is None or pdf_end is None:
        print("Could not extract dates from PDF, regenerating completely")
        return True, None, None

    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Check if PDF is up to date (end date should be at least yesterday)
    if pdf_end.date() >= yesterday.date():
        print(f"PDF is up to date (ends {pdf_end.date()}, yesterday was {yesterday.date()})")
        return False, None, None

    # PDF needs updating - determine missing date range
    missing_start = pdf_end + timedelta(days=1)  # Start from day after last PDF date

    # Load CSV to find actual last date in data
    df = pd.read_csv(csv_file)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    csv_end_date = df['timestamp_utc'].max()

    missing_end = csv_end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"PDF needs update: missing dates from {missing_start.date()} to {missing_end.date()}")

    return True, missing_start, missing_end


def generate_pages_for_date_range(csv_file, start_date, end_date,
                                   styles, anchor_img, sailing_img,
                                   cumulative_log, previous_day_log,
                                   last_pos_prev_day):
    """
    Generate story elements for a specific date range.

    Returns:
        tuple: (story_elements, new_cumulative_log, last_position)
    """
    # Load and filter data
    df = pd.read_csv(csv_file)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df['date'] = df['timestamp_utc'].dt.date
    df['time'] = df['timestamp_utc'].dt.strftime('%H:%M')  # ADD THIS LINE
    df = df.sort_values('timestamp_utc')

    # Filter for date range
    df_filtered = df[(df['timestamp_utc'].dt.date >= start_date.date()) &
                     (df['timestamp_utc'].dt.date <= end_date.date())]

    if len(df_filtered) == 0:
        print(f"No data found for date range {start_date.date()} to {end_date.date()}")
        return [], cumulative_log, last_pos_prev_day

    story = []
    grouped = df_filtered.groupby('date')

    for date, day_data in grouped:
        day_data = sample_hourly_data(day_data, max_entries=19)

        day_elements, cumulative_log = create_day_page(
            date, day_data, cumulative_log, previous_day_log,
            styles, anchor_img, sailing_img,
            last_pos_prev_day=last_pos_prev_day
        )

        story.extend(day_elements)
        previous_day_log = cumulative_log

        # Update last position
        last_pos_prev_day['lat'] = day_data.iloc[-1]['latitude']
        last_pos_prev_day['lon'] = day_data.iloc[-1]['longitude']

    return story, cumulative_log, last_pos_prev_day


def merge_pdfs(original_pdf, new_pages_pdf, output_pdf):
    """
    Merge original PDF (pages 3+) with new title page, track page, and new day pages.

    Args:
        original_pdf: Path to existing PDF
        new_pages_pdf: Path to PDF with new pages (title, track, new days)
        output_pdf: Path for output merged PDF
    """
    try:
        original = PdfReader(original_pdf)
        new_pages = PdfReader(new_pages_pdf)
        writer = PdfWriter()

        # Add new title page (page 0 from new_pages)
        writer.add_page(new_pages.pages[0])

        # Add new track page (page 1 from new_pages)
        writer.add_page(new_pages.pages[1])

        # Add original pages starting from page 2 (skip old title and track)
        for page_num in range(2, len(original.pages)):
            writer.add_page(original.pages[page_num])

        # Add new day pages (pages 2+ from new_pages)
        for page_num in range(2, len(new_pages.pages)):
            writer.add_page(new_pages.pages[page_num])

        # Write merged PDF
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)

        print(f"Successfully merged PDFs: {output_pdf}")
        return True

    except Exception as e:
        print(f"Error merging PDFs: {e}")
        return False


def generate_or_update_logbook_pdf(csv_file, output_pdf='logbook.pdf',
                                    anchor_img='big-anchor.png',
                                    sailing_img='sailing-boat.png'):
    """
    Generate new logbook PDF or update existing one with missing dates.
    """
    # Verify icon files
    if not os.path.exists(anchor_img):
        print(f"Warning: {anchor_img} not found. Status icons will be skipped.")
        anchor_img = None
    if not os.path.exists(sailing_img):
        print(f"Warning: {sailing_img} not found. Status icons will be skipped.")
        sailing_img = None

    # Check if update is needed
    needs_update_flag, missing_start, missing_end = needs_update(output_pdf, csv_file)

    if not needs_update_flag:
        print("PDF is already up to date. No changes needed.")
        return

    # Read full data
    df = pd.read_csv(csv_file)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df['date'] = df['timestamp_utc'].dt.date
    df['time'] = df['timestamp_utc'].dt.strftime('%H:%M')
    df = df.sort_values('timestamp_utc')

    styles = getSampleStyleSheet()

    # If PDF exists and we're updating (not creating new)
    if os.path.exists(output_pdf) and missing_start is not None:
        print("Updating existing PDF with new data...")

        # Calculate cumulative log up to the last day in existing PDF
        pdf_start, pdf_end = extract_date_range_from_pdf(output_pdf)
        df_old = df[df['timestamp_utc'].dt.date <= pdf_end.date()]

        cumulative_log = 0
        last_pos = {'lat': None, 'lon': None}

        for i in range(1, len(df_old)):
            dist = calculate_distance(
                df_old.iloc[i-1]['latitude'], df_old.iloc[i-1]['longitude'],
                df_old.iloc[i]['latitude'], df_old.iloc[i]['longitude']
            )
            cumulative_log += dist

        if len(df_old) > 0:
            last_pos['lat'] = df_old.iloc[-1]['latitude']
            last_pos['lon'] = df_old.iloc[-1]['longitude']

        previous_day_log = cumulative_log

        # Count existing day pages in original PDF (total pages - 2 for title/track)
        original_reader = PdfReader(output_pdf)
        existing_day_pages = len(original_reader.pages) - 2

        # Generate new pages (title, track, and missing days)
        temp_pdf = output_pdf.replace('.pdf', '_temp.pdf')

        pdf_temp = FooteredDocTemplate(
            temp_pdf,
            pagesize=landscape(A4),
            leftMargin=10*mm,
            rightMargin=10*mm,
            topMargin=15*mm,
            bottomMargin=20*mm
        )

        story = []

        # Add new title page with updated dates
        story.extend(create_title_page(styles, df))

        # Add updated track page
        if os.path.exists('ship_tracks.geojson'):
            story.extend(create_ship_track_page('ship_tracks.geojson'))

        # Generate pages for missing date range
        new_story, cumulative_log, last_pos = generate_pages_for_date_range(
            csv_file, missing_start, missing_end,
            styles, anchor_img, sailing_img,
            cumulative_log, previous_day_log, last_pos
        )

        story.extend(new_story)

        # Build temp PDF with page offset to continue numbering
        canvas_maker = make_canvas_with_offset(existing_day_pages)
        pdf_temp.build(story, canvasmaker=canvas_maker)

        # Merge: new title/track pages + old day pages + new day pages
        backup_pdf = output_pdf.replace('.pdf', '_backup.pdf')
        os.rename(output_pdf, backup_pdf)

        if merge_pdfs(backup_pdf, temp_pdf, output_pdf):
            print(f"Successfully updated: {output_pdf}")
            os.remove(backup_pdf)
            os.remove(temp_pdf)
        else:
            print("Merge failed, restoring backup")
            os.rename(backup_pdf, output_pdf)
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    else:
        # Generate complete new PDF
        print("Generating complete new logbook PDF...")

        pdf = FooteredDocTemplate(
            output_pdf,
            pagesize=landscape(A4),
            leftMargin=10*mm,
            rightMargin=10*mm,
            topMargin=15*mm,
            bottomMargin=20*mm
        )

        story = []

        # Add title page
        story.extend(create_title_page(styles, df))

        # Add ship track page
        if os.path.exists('ship_tracks.geojson'):
            story.extend(create_ship_track_page('ship_tracks.geojson'))

        # Process each day
        cumulative_log = 0
        previous_day_log = 0
        last_pos_prev_day = {'lat': None, 'lon': None}

        grouped = df.groupby('date')

        for date, day_data in grouped:
            day_data = sample_hourly_data(day_data, max_entries=19)

            day_elements, cumulative_log = create_day_page(
                date, day_data, cumulative_log, previous_day_log,
                styles, anchor_img, sailing_img,
                last_pos_prev_day=last_pos_prev_day
            )

            story.extend(day_elements)
            previous_day_log = cumulative_log

            last_pos_prev_day['lat'] = day_data.iloc[-1]['latitude']
            last_pos_prev_day['lon'] = day_data.iloc[-1]['longitude']

        # Build PDF with page offset 0 (starting from beginning)
        pdf.build(story, canvasmaker=NumberedCanvas)
        print(f"Logbook PDF generated: {output_pdf}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    generate_or_update_logbook_pdf('ais_position_reports.csv', 'nautical_logbook.pdf')
