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
from datetime import datetime
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

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import contextily as ctx
    HAS_GEO = True
except ImportError:
    print("Warning: geopandas/contextily not installed. Maps will be simplified.")
    HAS_GEO = False


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
COLUMN_WIDTHS = [10, 10, 10, 10, 11, 32, 10, 10, 11, 10, 10, 10, 20, 10, 38]

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

    Args:
        lat: Latitude
        lon: Longitude
        username: GeoNames username (register at geonames.org)

    Returns:
        Dictionary with location info or None if request fails
    """
    try:
        # Use findNearbyPlaceName for cities/towns
        url = "http://api.geonames.org/findNearbyPlaceNameJSON"
        params = {
            'lat': lat,
            'lng': lon,
            'username': username,
            'radius': 300,  # Search radius in km
            'maxRows': 1,
            'style': 'FULL'
        }

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200 and response.text != '{"geonames":[]}':  # Check for out of range empty response
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

        if response.status_code == 200 and response.text == '{"geonames":[]}':  # Check for out of range empty response
                    # Use findNearbyPlaceName for cities/towns
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
                    'feature':  '',
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
            return f"{distance:.1f} nm {bearing} of {formatted_location}"
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


def create_track_map(day_data):
    """Create track map with satellite basemap (if available)."""
    lats = day_data['latitude'].values
    lons = day_data['longitude'].values
    speeds = day_data['sog'].values

    # Fallback to simple plot if geo libraries not available
    if not HAS_GEO:
        return _create_simple_track_map(lons, lats)

    # Handle single point
    if len(lats) < 2:
        return _create_single_point_map(lats[0], lons[0])

    # Create map with track line
    return _create_geomap_with_track(lats, lons, speeds)


def _create_simple_track_map(lons, lats):
    """Simple fallback map without satellite imagery."""
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(lons, lats, 'r-', linewidth=2)
    ax.set_aspect('equal')
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf


def _create_single_point_map(lat, lon):
    """Create map for a single position (anchored vessel)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
    gdf_3857 = gdf.to_crs('EPSG:3857')

    # Plot point
    gdf_3857.plot(ax=ax, color='#00AA00', markersize=100,
                  edgecolor='white', linewidth=2, zorder=15, marker='o')

    # Set bounds with padding
    x, y = gdf_3857.geometry.iloc[0].x, gdf_3857.geometry.iloc[0].y
    padding = 1000
    ax.set_xlim(x - padding, x + padding)
    ax.set_ylim(y - padding, y + padding)
    ax.set_aspect('equal', adjustable='datalim')

        # Add basemap
    try:
        # 1. Base Layer: The high-resolution satellite imagery
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                    crs='EPSG:3857', zoom='auto', attribution=False)

        # 2. Top Layer: Just the labels (roads, cities, etc.)
        # This layer has a transparent background, creating the hybrid effect.
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronOnlyLabels,
                    crs='EPSG:3857', zoom='auto', attribution=False)
    except Exception as e:
        print(f"Could not load map tiles: {e}")
        ax.set_facecolor('#E8F4F8')
    scalebar = ScaleBar(1,  # 1 unit = 1 meter
                "m",  # Units are in meters
                location="lower left", # You can change this (e.g., 'lower right')
                frameon=False,         # No box around the scale bar
                color="white",         # Text color
                font_properties={"size": 10}
                    )
    ax.add_artist(scalebar)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf


def _create_geomap_with_track(lats, lons, speeds):
    """Create map with vessel track colored by speed."""
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import contextily as ctx
    from io import BytesIO
    from matplotlib_scalebar.scalebar import ScaleBar # Assuming this is available

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
    x_padded = x_range * 1.1  # <--- GEÄNDERT von 1.3 auf 1.1
    y_padded = y_range * 1.1  # <--- GEÄNDERT von 1.3 auf 1.1

    fig_aspect = 3.5 / 2.0  # Width/height
    data_aspect = x_padded / y_padded

    if data_aspect > fig_aspect:
        y_padded = x_padded / fig_aspect
    else:
        x_padded = y_padded * fig_aspect

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.set_xlim(x_center - x_padded/2, x_center + x_padded/2)
    ax.set_ylim(y_center - y_padded/2, y_center + y_padded/2)
    ax.set_aspect('equal', adjustable='datalim')

    # Plot track segments with speed-based coloring
    max_speed = max(speeds.max(), 1)
    for i in range(len(lons)-1):
        segment = LineString([(lons[i], lats[i]), (lons[i+1], lats[i+1])])
        gdf_seg = gpd.GeoDataFrame({'geometry': [segment]}, crs='EPSG:4326')
        gdf_seg_3857 = gdf_seg.to_crs('EPSG:3857')

        # Color based on speed (green=slow, yellow=medium, red=fast)
        speed_ratio = speeds[i] / max_speed
        if speed_ratio < 0.5:
            color = plt.cm.YlOrRd(speed_ratio * 2 * 0.3)
        else:
            color = plt.cm.YlOrRd(0.3 + (speed_ratio - 0.5) * 2 * 0.7)

        gdf_seg_3857.plot(ax=ax, color=color, linewidth=3, alpha=0.9, zorder=10)

        # Add basemap
    try:
        # 1. Base Layer: The high-resolution satellite imagery
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                    crs='EPSG:3857', zoom='auto', attribution=False)

        # 2. Top Layer: Just the labels (roads, cities, etc.)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronOnlyLabels,
                    crs='EPSG:3857', zoom='auto', attribution=False)
    except Exception as e:
        print(f"Could not load map tiles: {e}")
        ax.set_facecolor('#E8F4F8')
    scalebar = ScaleBar(1,  # 1 unit = 1 meter
                "m",  # Units are in meters
                location="lower left", # You can change this (e.g., 'lower right')
                frameon=False,         # No box around the scale bar
                color="white",         # Text color
                font_properties={"size": 10}
                    )
    ax.add_artist(scalebar)

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


# ============================================================================
# DATA PROCESSING
# ============================================================================

def sample_hourly_data(day_data, max_entries=19):
    """Sample data to fit on one page (max 19 entries + 3 summary rows)."""
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
        fmt(row.get('niederschlag'), '.1f'),
        cog_str,
        sog_str,
        f"{total_log:.1f}",
        format_position(row['latitude'], row['longitude'])[:30]
    ]


# ============================================================================
# PDF GENERATION
# ============================================================================

def create_title_page(styles):
    """Create title page elements."""
    title_bold = ParagraphStyle(
        'CustomTitle', parent=styles['Normal'],
        fontSize=36, leading=44, fontName='Helvetica-Bold', alignment=TA_CENTER
    )

    title_normal = ParagraphStyle(
        'CustomSubtitle', parent=styles['Normal'],
        fontSize=28, leading=34, fontName='Helvetica', alignment=TA_CENTER
    )

    ship_style = ParagraphStyle(
        'CustomShip', parent=styles['Normal'],
        fontSize=24, leading=30, fontName='Helvetica-Bold', alignment=TA_CENTER
    )

    elements = [
        Spacer(1, 80*mm),
        Paragraph("Nautic Horizons", title_bold),
        Spacer(1, 10*mm),
        Paragraph("2025/2026", title_normal),
        Spacer(1, 15*mm),
        Paragraph("Ship Regina Maris", ship_style),
        PageBreak()
    ]

    return elements


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
                    last_pos_prev_day={'lat': None, 'lon': None}): # <--- ADDED PARAMETER
    """Create a complete day page with header, charts, and logbook table."""
    elements = []

    # Date header
    # Set locale to German (Switzerland)
    locale.setlocale(locale.LC_TIME, 'de_CH.UTF-8')
    date_str = date.strftime('%A, %d. %B %Y')
    header = Paragraph(f"{date_str}", styles['Heading2'])
    elements.append(header)
    elements.append(Spacer(1, 2*mm))

    # Create charts
    pressure_chart = create_pressure_chart(day_data)
    track_map = create_track_map(day_data)

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

   # Create header table with charts - use exact data table width
    data_table_width = sum(COLUMN_WIDTHS)

    # Single row header: from_to on left, pressure chart, track map on right
    header_table = Table(
        [[from_to, pressure_img, track_img]],
        colWidths=[(data_table_width - 153)*mm, 60*mm, 93*mm]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),    # From/To text left-aligned
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),   # Pressure chart right-aligned
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),   # Track map right-aligned
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 3*mm))

    # Create logbook table
    table_data = [[
        'Zeit', 'Status', 'Wind\nR.', 'Wind\nKN', 'Wetter', 'Witterung',
        'mbar', 'T.Luft\n°C', 'T.Wasser\n°C', 'Wellen\nm', 'Regen\nmm',
        'Kurs', 'Fahrt', 'Log', 'Position'
    ]]

    # Add data rows
    day_log = 0  # Distance covered during this day only


    # Calculate distance from previous day's last position to current day's first position
    if last_pos_prev_day['lat'] is not None and len(day_data) > 0:
        dist_offset = calculate_distance(
            last_pos_prev_day['lat'], last_pos_prev_day['lon'],
            day_data.iloc[0]['latitude'], day_data.iloc[0]['longitude']
        )
        day_log += dist_offset


    for idx, row in day_data.iterrows():
        row_position = day_data.index.get_loc(idx)

        # Skip distance calculation for the very first point of the day
        # (The distance from the previous day's last point is already added above)
        if row_position > 0:
            prev_idx = day_data.index[row_position - 1]
            prev = day_data.loc[prev_idx]
            dist = calculate_distance(prev['latitude'], prev['longitude'],
                                    row['latitude'], row['longitude'])
            day_log += dist
        # The first position of the day's distance is accounted for by dist_offset above.

        table_data.append(format_table_row(row, day_data, day_log,
                                          anchor_img, sailing_img))

    # Add summary rows
    # day_log now contains the total distance for this day, including the offset
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      'Tagessumme', f'{day_log:.1f}', ''])
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      'Vortrag', f'{previous_day_log:.1f}', ''])

    cumulative_log += day_log
    table_data.append(['', '', '', '', '', '', '', '', '', '', '', '',
                      'Gesamt', f'{cumulative_log:.1f}', ''])

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


def generate_logbook_pdf(csv_file, output_pdf='logbook.pdf',
                         anchor_img='big-anchor.png',
                         sailing_img='sailing-boat.png'):
    """
    Generate nautical logbook PDF from AIS position reports CSV.

    Args:
        csv_file: Path to CSV file with AIS position data
        output_pdf: Output PDF filename (default: 'logbook.pdf')
        anchor_img: Path to anchor icon image
        sailing_img: Path to sailing boat icon image
    """
    # Verify icon files
    if not os.path.exists(anchor_img):
        print(f"Warning: {anchor_img} not found. Status icons will be skipped.")
        anchor_img = None
    if not os.path.exists(sailing_img):
        print(f"Warning: {sailing_img} not found. Status icons will be skipped.")
        sailing_img = None

    # Read and prepare data
    df = pd.read_csv(csv_file)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df['date'] = df['timestamp_utc'].dt.date
    df['time'] = df['timestamp_utc'].dt.strftime('%H:%M')
    df = df.sort_values('timestamp_utc')

    # Create PDF document
    pdf = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=10*mm,
        rightMargin=10*mm,
        topMargin=15*mm,
        bottomMargin=15*mm
    )

    story = []
    styles = getSampleStyleSheet()

    # Add title page
    story.extend(create_title_page(styles))

    # Process each day
    cumulative_log = 0
    previous_day_log = 0
    # Store the last position of the *previous* day for continuity
    last_pos_prev_day = {'lat': None, 'lon': None}

    grouped = df.groupby('date')

    for date, day_data in grouped:
        # Sample data to fit on one page
        day_data = sample_hourly_data(day_data, max_entries=19)

        # Create day page, passing the last position of the previous day
        day_elements, cumulative_log = create_day_page(
            date, day_data, cumulative_log, previous_day_log,
            styles, anchor_img, sailing_img,
            last_pos_prev_day=last_pos_prev_day # <--- NEW PARAMETER
        )

        story.extend(day_elements)
        previous_day_log = cumulative_log

        # Update last_pos_prev_day with the *current* day's last position
        last_pos_prev_day['lat'] = day_data.iloc[-1]['latitude']
        last_pos_prev_day['lon'] = day_data.iloc[-1]['longitude']


    # Build PDF
    pdf.build(story)
    print(f"Logbook PDF generated: {output_pdf}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    generate_logbook_pdf('ais_position_reports.csv', 'nautical_logbook.pdf')