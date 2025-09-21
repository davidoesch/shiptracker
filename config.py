# AIS Stream Configuration
# Configuration file for AIS stream monitoring parameters

# Ship MMSI filters - list of MMSI numbers to monitor
# Example: ["244528000", "123456789", "987654321"]
# REGINA MARIS is: "244528000"
# Blue Satr ferry is: "241087000"
FILTERS_SHIP_MMSI = ["241087000"]

# Maximum monitoring duration in minutes
# How long to monitor the AIS stream before stopping
MAX_DURATION_MINUTES = 10

# Time threshold in hours for duplicate entry filtering
# Skip new entries if the latest entry is younger than this threshold
TIME_THRESHOLD_HOURS = 0.1