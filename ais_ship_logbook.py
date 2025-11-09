"""
Nautical Logbook PDF Generator
Generates daily logbook entries from AIS position reports CSV with charts and maps
"""

import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak, Spacer, Image as RLImage
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Circle, Line, Polygon, Wedge
from reportlab.graphics import renderPDF
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MPLCircle
from io import BytesIO
import numpy as np
import os

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except ImportError:
    print("Warning: geopandas not installed. Install with: pip install geopandas")
    gpd = None

try:
    import contextily as ctx
except ImportError:
    print("Warning: contextily not installed. Install with: pip install contextily")
    ctx = None


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in nautical miles"""
    R = 3440.065  # Earth radius in nautical miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def format_position(lat, lon):
    """Format coordinates in degrees and decimal minutes"""
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_dir = 'N' if lat >= 0 else 'S'

    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_dir = 'E' if lon >= 0 else 'W'

    return f"{lat_deg}° {lat_min:.3f}' {lat_dir} {lon_deg}° {lon_min:.3f}' {lon_dir}"


def get_wind_direction_abbreviation(degrees):
    """Convert wind direction degrees to abbreviation"""
    if pd.isna(degrees):
        return "---"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((degrees + 11.25) / 22.5)
    return dirs[ix % 16]


def create_wind_barb(wind_speed, wind_dir):
    """Create wind barb symbol - reduced by 20%"""
    d = Drawing(9.6, 9.6)  # Reduced from 12 to 9.6

    if pd.isna(wind_speed) or pd.isna(wind_dir) or wind_speed == 0:
        return d

    # Convert to radians (meteorological: direction wind comes FROM)
    rad = math.radians(wind_dir + 180)  # Add 180 to show where wind goes TO

    # Staff line - reduced by 20%
    staff_len = 6.0  # Reduced from 7.5
    x1, y1 = 4.8, 4.8  # Reduced from 6, 6
    x2 = x1 + staff_len * math.sin(rad)
    y2 = y1 + staff_len * math.cos(rad)

    d.add(Line(x1, y1, x2, y2, strokeWidth=0.72))  # Reduced from 0.9

    # Add barbs based on wind speed (5 knots per half barb, 10 per full barb)
    knots = wind_speed
    full_barbs = int(knots / 10)
    half_barbs = 1 if (knots % 10) >= 5 else 0

    # Position barbs along staff - reduced by 20%
    barb_spacing = 1.52  # Reduced from 1.9
    current_pos = 0

    for i in range(full_barbs):
        bx = x2 - current_pos * math.sin(rad)
        by = y2 - current_pos * math.cos(rad)

        # Full barb (perpendicular line) - reduced by 20%
        perp_rad = rad + math.radians(90)
        barb_len = 2.08  # Reduced from 2.6
        bx2 = bx + barb_len * math.sin(perp_rad)
        by2 = by + barb_len * math.cos(perp_rad)

        d.add(Line(bx, by, bx2, by2, strokeWidth=0.72))  # Reduced from 0.9
        current_pos += barb_spacing

    if half_barbs:
        bx = x2 - current_pos * math.sin(rad)
        by = y2 - current_pos * math.cos(rad)

        perp_rad = rad + math.radians(90)
        barb_len = 1.04  # Reduced from 1.3
        bx2 = bx + barb_len * math.sin(perp_rad)
        by2 = by + barb_len * math.cos(perp_rad)

        d.add(Line(bx, by, bx2, by2, strokeWidth=0.72))  # Reduced from 0.9

    return d


def create_cloud_cover(cloud_cover):
    """Create cloud cover circle symbol - reduced by 20%"""
    d = Drawing(9.6, 9.6)  # Reduced from 12 to 9.6

    # Cloud cover circle - reduced by 20%
    circle = Circle(4.8, 4.8, 3.6)  # Reduced from (6, 6, 4.5)
    circle.strokeWidth = 0.72  # Reduced from 0.9
    circle.strokeColor = colors.black

    # Determine fill based on cloud cover
    if pd.isna(cloud_cover):
        circle.fillColor = colors.white
        d.add(circle)
        return d

    if cloud_cover < 25:
        circle.fillColor = colors.white
        d.add(circle)
    elif cloud_cover < 50:
        # Quarter filled - add circle first, then wedge on top
        circle.fillColor = colors.white
        d.add(circle)
        # Create wedge for quarter fill (positioned at same center as circle)
        wedge = Wedge(4.8, 4.8, 3.6, 90, 180, fillColor=colors.black, strokeColor=None)
        d.add(wedge)
    elif cloud_cover < 75:
        # Half filled - add circle first, then wedge on top
        circle.fillColor = colors.white
        d.add(circle)
        wedge = Wedge(4.8, 4.8, 3.6, 0, 180, fillColor=colors.black, strokeColor=None)
        d.add(wedge)
    else:
        # Fully filled
        circle.fillColor = colors.black
        d.add(circle)

    return d


def create_weather_cell(wind_speed, wind_dir, cloud_cover):
    """Create combined cell with wind barb and cloud cover side by side - reduced by 20%"""
    d = Drawing(19.2, 9.6)  # Reduced from (24, 12)

    # Wind barb on left
    wind_barb = create_wind_barb(wind_speed, wind_dir)
    for item in wind_barb.contents:
        d.add(item)

    # Cloud cover on right (shifted by 9.6 pixels, reduced from 12)
    cloud = create_cloud_cover(cloud_cover)
    for item in cloud.contents:
        # Shift x coordinate by 9.6 (reduced from 12)
        if hasattr(item, 'x'):
            item.x += 9.6
        elif hasattr(item, 'x1'):
            item.x1 += 9.6
            item.x2 += 9.6
        if hasattr(item, 'cx'):
            item.cx += 9.6
        # Also shift wedge coordinates for cloud fill
        if isinstance(item, Wedge):
            item.centerx += 9.6
        d.add(item)

    return d


def sample_hourly_data(day_data, max_entries=19):
    """Sample data to fit on one page (max 19 entries + 3 summary rows = 22 total rows)"""
    if len(day_data) <= max_entries:
        return day_data.reset_index(drop=True)

    # Keep first and last
    first = day_data.iloc[0:1].copy()
    last = day_data.iloc[-1:].copy()
    middle = day_data.iloc[1:-1].copy()

    # Sample middle data by hour
    middle.loc[:, 'hour'] = pd.to_datetime(middle['timestamp_utc']).dt.hour
    sampled_middle = middle.groupby('hour').first().reset_index(drop=True)

    # Combine and sort
    result = pd.concat([first, sampled_middle, last]).drop_duplicates(subset=['timestamp_utc'])
    result = result.sort_values('timestamp_utc').reset_index(drop=True)

    # Limit to max_entries
    if len(result) > max_entries:
        # Keep first, last, and evenly spaced middle entries
        indices = [0] + list(np.linspace(1, len(result)-2, max_entries-2, dtype=int)) + [len(result)-1]
        result = result.iloc[indices].copy()

    return result.reset_index(drop=True)


def create_pressure_chart(day_data):
    """Create pressure chart for the day with 0-24 hour range"""
    fig, ax = plt.subplots(figsize=(2.8, 2.0))

    # Parse times and get pressures
    times_str = day_data['time'].values
    pressures = day_data['luftdruck'].dropna().values
    time_objs = [datetime.strptime(t, '%H:%M') for t in times_str[:len(pressures)]]
    time_hours = [t.hour + t.minute/60 for t in time_objs]

    if len(pressures) > 0:
        # Plot without fill
        ax.plot(time_hours, pressures, 'k-', linewidth=2, marker='o', markersize=3)

        # Y-axis: 8 ticks from min to max
        y_min = pressures.min()
        y_max = pressures.max()
        y_ticks = np.linspace(y_min, y_max, 8)
        ax.set_ylim(y_min - 0.2, y_max + 0.2)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=7)
        ax.set_ylabel('mbar', fontsize=8)

        # X-axis: 0-24 hours, show only 06:00, 12:00, 18:00
        ax.set_xlim(0, 24)
        ax.set_xticks([6, 12, 18])
        ax.set_xticklabels(['06:00', '12:00', '18:00'], fontsize=8)

        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def get_image_with_aspect(img_buffer, max_width_mm, max_height_mm):
    """Create ReportLab Image with preserved aspect ratio

    Args:
        img_buffer: BytesIO buffer containing image
        max_width_mm: Maximum width in mm
        max_height_mm: Maximum height in mm

    Returns:
        RLImage object with correct dimensions
    """
    from reportlab.lib import utils

    # Get actual image size
    img = utils.ImageReader(img_buffer)
    iw, ih = img.getSize()
    aspect = ih / float(iw)

    # Calculate dimensions to fit within max bounds
    width_mm = max_width_mm
    height_mm = width_mm * aspect

    # If height exceeds max, scale by height instead
    if height_mm > max_height_mm:
        height_mm = max_height_mm
        width_mm = height_mm / aspect

    return RLImage(img_buffer, width=width_mm*mm, height=height_mm*mm)


def create_track_map(day_data):
    """Create track map using geopandas and contextily ESRI basemap - fixed distortion"""

    if gpd is None or ctx is None:
        # Fallback to simple plot if libraries not available
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        ax.plot(day_data['longitude'], day_data['latitude'], 'r-', linewidth=2)
        ax.set_aspect('equal')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        return buf

    lats = day_data['latitude'].values
    lons = day_data['longitude'].values
    speeds = day_data['sog'].values

    # Handle case with only one point
    if len(lats) < 2:
        fig, ax = plt.subplots(figsize=(3.5, 2.0))

        # Create single point GeoDataFrame
        point = Point(lons[0], lats[0])
        gdf_point = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
        gdf_point_3857 = gdf_point.to_crs('EPSG:3857')

        # Plot the single point
        gdf_point_3857.plot(ax=ax, color='#00AA00', markersize=100,
                           edgecolor='white', linewidth=2, zorder=15, marker='o')

        # Set bounds with padding around single point
        x, y = gdf_point_3857.geometry.iloc[0].x, gdf_point_3857.geometry.iloc[0].y
        padding = 2500  # 2.5 km padding
        ax.set_xlim(x - padding, x + padding)
        ax.set_ylim(y - padding, y + padding)

        # Add basemap
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                           crs='EPSG:3857', alpha=0.9, zoom='auto', attribution=False)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                           crs='EPSG:3857', alpha=0.3, zoom='auto', attribution=False)
        except Exception as e:
            print(f"Could not load map tiles: {e}")
            ax.set_facecolor('#E8F4F8')

        ax.axis('off')
        plt.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        return buf

    # Create GeoDataFrame with LineString geometry
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    line = LineString([(lon, lat) for lon, lat in zip(lons, lats)])

    # Create GeoDataFrame for the track line
    gdf_line = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:4326')

    # Create GeoDataFrame for points (start/end markers)
    gdf_points = gpd.GeoDataFrame({
        'type': ['start', 'end'],
        'geometry': [points[0], points[-1]]
    }, crs='EPSG:4326')

    # Convert to Web Mercator (EPSG:3857) for contextily
    gdf_line_3857 = gdf_line.to_crs('EPSG:3857')
    gdf_points_3857 = gdf_points.to_crs('EPSG:3857')

    # Get bounds in Web Mercator
    xmin, ymin, xmax, ymax = gdf_line_3857.total_bounds

    # Calculate ranges
    x_range = max(xmax - xmin, 100)  # minimum 100m
    y_range = max(ymax - ymin, 100)

    # Calculate centers
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    # Add 30% padding to both dimensions independently
    x_padded = x_range * 1.3
    y_padded = y_range * 1.3

    # Calculate aspect ratio of the figure (width/height)
    fig_aspect = 3.5 / 2.0  # 1.75

    # Calculate aspect ratio of the data extent
    data_aspect = x_padded / y_padded

    # Adjust extents to match figure aspect ratio without distortion
    if data_aspect > fig_aspect:
        # Data is wider than figure - expand y to match
        y_padded = x_padded / fig_aspect
    else:
        # Data is taller than figure - expand x to match
        x_padded = y_padded * fig_aspect

    # Create figure with exact aspect ratio
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    # Set bounds with corrected aspect ratio
    ax.set_xlim(x_center - x_padded/2, x_center + x_padded/2)
    ax.set_ylim(y_center - y_padded/2, y_center + y_padded/2)

    # CRITICAL: Set aspect to 'equal' to prevent distortion
    ax.set_aspect('equal', adjustable='datalim')

    # Plot track with speed-based coloring
    max_speed = max(speeds.max(), 1)

    # Convert individual segments to Web Mercator for coloring
    for i in range(len(lons)-1):
        segment = LineString([(lons[i], lats[i]), (lons[i+1], lats[i+1])])
        gdf_segment = gpd.GeoDataFrame({'geometry': [segment]}, crs='EPSG:4326').to_crs('EPSG:3857')

        speed_ratio = speeds[i] / max_speed
        # Green (slow) to Yellow to Red (fast)
        if speed_ratio < 0.5:
            color = plt.cm.YlOrRd(speed_ratio * 2 * 0.3)
        else:
            color = plt.cm.YlOrRd(0.3 + (speed_ratio - 0.5) * 2 * 0.7)

        gdf_segment.plot(ax=ax, color=color, linewidth=3, alpha=0.9, zorder=10)

    # Add ESRI basemaps
    try:
        # Add satellite imagery
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                       crs='EPSG:3857', alpha=0.9, zoom='auto', attribution=False)
        # Add labels/boundaries overlay
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                       crs='EPSG:3857', alpha=0.3, zoom='auto', attribution=False)
    except Exception as e:
        print(f"Could not load map tiles: {e}")
        ax.set_facecolor('#E8F4F8')

    # Add start and end markers
    start_point = gdf_points_3857[gdf_points_3857['type'] == 'start']
    end_point = gdf_points_3857[gdf_points_3857['type'] == 'end']

    start_point.plot(ax=ax, color='#00AA00', markersize=60,
                     edgecolor='white', linewidth=1.5, zorder=15, marker='o')
    end_point.plot(ax=ax, color='#CC0000', markersize=80,
                   edgecolor='white', linewidth=1.5, zorder=15, marker='^')

    # Remove axis
    ax.axis('off')

    plt.tight_layout(pad=0)

    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf




def generate_logbook_pdf(csv_file, output_pdf='logbook.pdf', anchor_img='big-anchor.png', sailing_img='sailing-boat.png'):
    """
    Generate nautical logbook PDF from AIS position reports CSV

    Args:
        csv_file: Path to the CSV file with AIS position data
        output_pdf: Output PDF filename (default: 'logbook.pdf')
        anchor_img: Path to anchor icon image
        sailing_img: Path to sailing boat icon image
    """
    # Check if icon files exist
    if not os.path.exists(anchor_img):
        print(f"Warning: {anchor_img} not found. Status icons will be skipped.")
        anchor_img = None
    if not os.path.exists(sailing_img):
        print(f"Warning: {sailing_img} not found. Status icons will be skipped.")
        sailing_img = None

    # Read CSV
    df = pd.read_csv(csv_file)

    # Convert to datetime and convert to UTC
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    df['date'] = df['timestamp_utc'].dt.date
    df['time'] = df['timestamp_utc'].dt.strftime('%H:%M')

    # Sort by timestamp
    df = df.sort_values('timestamp_utc')

    # Create PDF
    page_width, page_height = landscape(A4)
    pdf = SimpleDocTemplate(output_pdf, pagesize=landscape(A4),
                        leftMargin=10*mm, rightMargin=10*mm,
                        topMargin=15*mm, bottomMargin=15*mm)

    story = []
    styles = getSampleStyleSheet()

    # Add title page to story
    # Create custom styles matching the canvas version
    title_style_bold = ParagraphStyle(
        'CustomTitle',
        parent=styles['Normal'],
        fontSize=36,
        leading=44,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    title_style_normal = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=28,
        leading=34,
        fontName='Helvetica',
        alignment=TA_CENTER
    )

    ship_style = ParagraphStyle(
        'CustomShip',
        parent=styles['Normal'],
        fontSize=24,
        leading=30,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    # Add title page content with proper vertical centering
    story.append(Spacer(1, 80*mm))  # Vertical centering
    story.append(Paragraph("Nautic Horizons", title_style_bold))
    story.append(Spacer(1, 10*mm))  # Space between title and year
    story.append(Paragraph("2025/2026", title_style_normal))
    story.append(Spacer(1, 15*mm))  # Slightly more space before ship name
    story.append(Paragraph("Ship Regina Maris", ship_style))
    story.append(PageBreak())


    # Group by date
    grouped = df.groupby('date')

    # Track cumulative log
    cumulative_log = 0
    previous_day_log = 0

    for date_idx, (date, day_data) in enumerate(grouped):
        # Sample to fit on one page (19 data rows + 3 summary rows = 22 total rows)
        day_data = sample_hourly_data(day_data, max_entries=19)

        # Create header with date
        date_str = date.strftime('%A, %d. %B %Y')
        header = Paragraph(f"<b>DATUM</b> {date_str}", styles['Heading2'])
        story.append(header)
        story.append(Spacer(1, 2*mm))

        # Create charts side by side
        pressure_chart = create_pressure_chart(day_data)
        track_map = create_track_map(day_data)

        pressure_img = get_image_with_aspect(pressure_chart, 60, 43)
        track_img = get_image_with_aspect(track_map, 93, 40)

        # Header with position info and charts
        first_pos = format_position(day_data.iloc[0]['latitude'], day_data.iloc[0]['longitude'])
        last_pos = format_position(day_data.iloc[-1]['latitude'], day_data.iloc[-1]['longitude'])

        from_to_text = f"<b>VON</b> {first_pos}<br/><b>NACH</b> {last_pos}"
        from_to = Paragraph(from_to_text, styles['Normal'])

        # Create header layout: text on left, charts on right side by side
        chart_data = [[pressure_img, track_img]]
        chart_table = Table(chart_data, colWidths=[60*mm, 93*mm])
        chart_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),   # Pressure chart left-aligned in its cell
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),   # Track map left-aligned in its cell
        ]))

        # Create table - adjusted column widths to fit on one page
        col_widths = [10*mm, 10*mm, 10*mm, 10*mm, 11*mm, 32*mm, 10*mm, 10*mm, 11*mm, 10*mm, 10*mm, 10*mm, 20*mm, 10*mm, 38*mm]

        # Calculate the total width of the main data table
        data_table_width = sum(col_widths)  # This equals 232mm
        chart_table_width = 60*mm + 93*mm  # 130mm

        # Adjust header table widths to align right edge
        from_to_width = data_table_width - chart_table_width

        header_data = [[from_to, chart_table]]
        header_table = Table(header_data, colWidths=[from_to_width, chart_table_width])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),  # Align charts to the right
        ]))

        story.append(header_table)
        story.append(Spacer(1, 3*mm))

        # Create main logbook table
        table_data = [[
            'Zeit', 'Status', 'Wind\nR.', 'Wind\nKN', 'Wetter', 'Witterung',
            'mbar', 'T.Luft\n°C', 'T.Wasser\n°C', 'Wellen\nm', 'Regen\nmm',
            'Kurs', 'Fahrt', 'Log', 'Position'
        ]]

        # Calculate daily totals
        total_distance = 0
        total_log = 0

        for idx, row in day_data.iterrows():
            # Calculate distance from previous point in the sampled data
            row_position = day_data.index.get_loc(idx)

            if row_position > 0:
                prev_idx = day_data.index[row_position - 1]
                prev = day_data.loc[prev_idx]
                dist = calculate_distance(prev['latitude'], prev['longitude'],
                                        row['latitude'], row['longitude'])
                total_distance += dist
            else:
                dist = 0

            total_log += dist

            # Determine status icon - reduced by 20%
            nav_status = row.get('navigational_status', 0)
            sog = row.get('sog', 0)

            # Check for no data values
            if pd.isna(sog) or sog == 999:
                sog = 0

            status_icon = None
            if (nav_status == 1 or sog == 0) and anchor_img:
                status_icon = RLImage(anchor_img, width=3.2*mm, height=3.2*mm)  # Reduced from 4mm
            elif sog > 0 and sailing_img:
                status_icon = RLImage(sailing_img, width=3.2*mm, height=3.2*mm)  # Reduced from 4mm

            # Get wind direction abbreviation
            wind_dir_abbr = get_wind_direction_abbreviation(row.get('windrichtung'))
            wind_kn = f"{row['windstaerke']:.1f}" if pd.notna(row.get('windstaerke')) else "---"

            # Create weather cell (wind barb + cloud cover side by side)
            weather_cell = create_weather_cell(
                row.get('windstaerke'),
                row.get('windrichtung'),
                row.get('bewoelkung')
            )

            # Weather condition (Witterung) - doubled width column
            witterung = row.get('wetterzustand', '')
            if pd.notna(witterung) and len(str(witterung)) > 24:
                witterung = str(witterung)[:24]
            elif pd.isna(witterung):
                witterung = ''

            # Handle COG
            cog = row.get('cog')
            if pd.isna(cog) or cog == 999 or cog == 360:
                cog_str = "---"
            else:
                cog_str = f"{cog:.0f}°"

            # Handle SOG
            if sog == 0:
                sog_str = "0.0"
            else:
                sog_str = f"{sog:.1f}"

            # Additional weather data
            temp_air = f"{row['lufttemperatur']:.1f}" if pd.notna(row.get('lufttemperatur')) else "---"
            temp_water = f"{row['wassertemperatur']:.1f}" if pd.notna(row.get('wassertemperatur')) else "---"
            waves = f"{row['wellenhoehe']:.2f}" if pd.notna(row.get('wellenhoehe')) else "---"
            rain = f"{row['niederschlag']:.1f}" if pd.notna(row.get('niederschlag')) else "---"

            table_data.append([
                row['time'],
                status_icon if status_icon else "",
                wind_dir_abbr,
                wind_kn,
                weather_cell,
                witterung,
                f"{row['luftdruck']:.1f}" if pd.notna(row.get('luftdruck')) else '',
                temp_air,
                temp_water,
                waves,
                rain,
                cog_str,
                sog_str,
                f"{total_log:.1f}",
                format_position(row['latitude'], row['longitude'])[:30]
            ])

        # Update cumulative totals
        day_log = total_log

        # Add summary rows
        table_data.append(['', '', '', '', '', '', '', '', '', '', '', '', 'Tagessumme', f'{day_log:.1f}', ''])
        table_data.append(['', '', '', '', '', '', '', '', '', '', '', '', 'Vortrag', f'{previous_day_log:.1f}', ''])

        cumulative_log += day_log
        table_data.append(['', '', '', '', '', '', '', '', '', '', '', '', 'Gesamt', f'{cumulative_log:.1f}', ''])

        # Update for next day
        previous_day_log = cumulative_log



        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 0.5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 0.5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, -3), (-1, -1), colors.lightgrey),
            ('FONTNAME', (0, -3), (-1, -1), 'Helvetica-Bold'),
        ]))

        story.append(table)
        story.append(PageBreak())

    # Build PDF
    pdf.build(story)
    print(f"Logbook PDF generated: {output_pdf}")


# Example usage
if __name__ == "__main__":
    generate_logbook_pdf('ais_position_reports.csv', 'nautical_logbook.pdf')
