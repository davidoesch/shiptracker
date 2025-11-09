# AIS Stream Configuration
# Configuration file for AIS stream monitoring parameters

# Ship MMSI filters - list of MMSI numbers to monitor
# Example: ["244528000", "123456789", "987654321"]
# REGINA MARIS is: "244528000"
FILTERS_SHIP_MMSI_ID = ["244528000","268855"]


# Blue Star ferry is: "241087000"
#FILTERS_SHIP_MMSI = ["241087000"]

# Yacht : "241087000"
#FILTERS_SHIP_MMSI = ["247360450"]

# Maximum monitoring duration in minutes
# How long to monitor the AIS stream before stopping
MAX_DURATION_MINUTES = 0.001

# Time threshold in hours for duplicate entry filtering
# Skip new entries if the latest entry is younger than this threshold
TIME_THRESHOLD_HOURS = 0.15

# Bounding boxes for geographical filtering
# Each bounding box is defined by two coordinates: [[lat1, lon1], [lat2, lon2]]
# Example: [[[34, 18], [42, 30]]] Aegean Sea

#"BoundingBoxes": [[[-90, -180], [90, 180]]], # Worldwide

# Approximate bounding box for the Aegean Sea
#BOUNDING_BOXES = [[[34, 18], [42, 30]]]

# North Atlantic Ocean ande carribean sea
BOUNDING_BOXES = [[[-7.22,-105.15], [61.77,14.77]]]
