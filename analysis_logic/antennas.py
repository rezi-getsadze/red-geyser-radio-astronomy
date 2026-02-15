import math
import folium

# Constants for Earth
EARTH_RADIUS = 6371.0  # in km
EARTH_ECCENTRICITY = 0.0818191908426

# Function to convert ECEF to lat/lon/alt
def ecef_to_geodetic(x, y, z):
    # Compute longitude
    lon = math.atan2(y, x)
    
    # Compute initial latitude
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - EARTH_ECCENTRICITY**2))
    
    # Iterate to improve the latitude estimate
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = EARTH_RADIUS / math.sqrt(1 - EARTH_ECCENTRICITY**2 * sin_lat**2)
        alt = p / math.cos(lat) - N
        lat = math.atan2(z + N * EARTH_ECCENTRICITY**2 * sin_lat, p)
    
    # Convert radians to degrees
    lon = math.degrees(lon)
    lat = math.degrees(lat)
    
    return lat, lon, alt

# ECEF coordinates for VLBA antennas
antenna_coords = [
    (-2112065.3493, -3705356.5139, 4726813.5909),  # BR
    (-1324009.4471, -5332181.9564, 3231962.3351),  # FD
    (1446374.7134, -4447939.6992, 4322306.2177),   # HN
    (-1449752.7211, -4975298.5765, 3709123.7824),  # LA
    (-5464075.3127, -2495247.5080, 2148297.6506),  # MK
    (-130872.6473, -4762317.0871, 4226850.9707),   # NL
    (-2409150.5789, -4478573.0616, 3838617.2889),  # OV
    (-1640954.0758, -5014816.0284, 3575411.7209),  # PT
    (2607848.7200, -5488069.4542, 1932739.8517),   # SC
]

# Create a map centered at one of the antenna locations
bx, by, bz = antenna_coords[0]  # Use Brewster as the center point
lat, lon, _ = ecef_to_geodetic(bx, by, bz)  # Convert to lat/lon

# Create the map with a suitable zoom level
mymap = folium.Map(location=[lat, lon], zoom_start=4)

# Define which antennas should have different colors
red_antennas = [4]  # Antenna 5
black_antennas = [8]  # Antenna 10

# Add antenna locations to the map
for index, (bx, by, bz) in enumerate(antenna_coords):
    lat, lon, _ = ecef_to_geodetic(bx, by, bz)  # Convert to lat/lon
    
    # Determine the correct color
    if index in red_antennas:
        color = "red"
    elif index in black_antennas:
        color = "black"
    else:
        color = "blue"
    
    folium.Marker(
        [lat, lon], 
        tooltip=f"Antenna {index + 1} ({lat:.2f}, {lon:.2f})",
        icon=folium.Icon(color=color)  # Set marker color
    ).add_to(mymap)  # Add markers to the map

# Save the map to an HTML file
mymap.save("vlba_antennas.html")
