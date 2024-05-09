#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:24:36 2024

@author: thomasminahan
"""

import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

# Initialize logging
# logging.basicConfig(level=logging.INFO)

# Ship Speed DataFrame
shipclasses = pd.DataFrame({
    'class': ['Sloop', 'Brig', 'Gally'],
    'full_speed': [40, 4, 3],  # Speeds in seconds/unit
    'partial_speed': [45, 2, 1],
    'no_wind': [50, 0.5, 0.3]
})

# Ship Type Mapping
Chosen_SHIP = {"S": 1, "B": 2, "G": 3}

# Wind Directions
wind_directions = {
    "E": 0, "SSE": 22.5, "SE": 45, "ESE": 67.5, "S": 90, "SWS": 112.5, "SW": 135, "WSW": 157.5,
    "W": 180, "WNW": 202.5, "NW": 225, "NNW": 247.5, "N": 270, "NNE": 292.5, "NE": 315, "ENE": 337.5
}

# Destination Locations
Destinations = {
"outposts": [
        ("Sanctuary Outpost", ("F", 7)),
        ("Golden Sands Outpost", ("E", 12)),
        ("Plunder Outpost", ("J", 18)),
        ("Ancient Spire Outpost", ("Q", 17)),
        ("Galleon's Grave Outpost", ("R", 8)),
        ("Dagger Tooth Outpost", ("M", 8)),
        ("Morrow's Peak Outpost", ("V", 17)),
        ("Port Merrick", ("D",10))
    ],
"Any Fort": [
        ("Royal Crest Fortress", ("J", 6)),
        ("Imperial Crown Fortress", ("B", 11)),
        ("Ancient Gold Fortress", ("F", 19)),
        ("Old Brinestone Fortress", ("K", 21)),
        ("Traitor's Fate Fortress", ("S", 6)),
        ("Mercy's End Fortress", ("P", 14)),
        ("Keel Haul Fort", ("C", 6)),
        ("Hidden Spring Keep", ("I", 8)),
        ("Sailor's Knot Stronghold", ("E", 14)),
        ("Lost Gold Fort", ("H", 17)),
        ("Fort of the Damned", ("L", 14)),
        ("The Crow's Nest Fortress", ("O", 17)),
        ("Skull Keep", ("P", 9)),
        ("Kraken Watchtower", ("L", 6)),
        ("Shark Fin Camp", ("P", 5)),
        ("Molten Sands Fortress", ("Z", 11))
    ],
"sea forts": [
        ("Royal Crest Fortress", ("J", 6)),
        ("Imperial Crown Fortress", ("B", 11)),
        ("Ancient Gold Fortress", ("F", 19)),
        ("Old Brinestone Fortress", ("K", 21)),
        ("Traitor's Fate Fortress", ("S", 6)),
        ("Mercy's End Fortress", ("P", 14))
    ],
"skeleton forts": [
        ("Keel Haul Fort", ("C", 6)),
        ("Hidden Spring Keep", ("I", 8)),
        ("Sailor's Knot Stronghold", ("E", 14)),
        ("Lost Gold Fort", ("H", 17)),
        ("Fort of the Damned", ("L", 14)),
        ("The Crow's Nest Fortress", ("O", 17)),
        ("Skull Keep", ("P", 9)),
        ("Kraken Watchtower", ("L", 6)),
        ("Shark Fin Camp", ("P", 5)),
        ("Molten Sands Fortress", ("Z", 11))
    ],
"seaports": [
    ("The Spoils of Plenty Store", ("B", 7)),
    ("The North Star Seapost", ("H", 10)),
    ("The Finest Trading Post", ("F", 17)),
    ("Stephen's Spoils", ("L", 15)),
    ("Three Paces East Seapost", ("S", 10)),
    ("The Wild Treasures Store", ("O", 4)),
    ("Brian's Bazaar", ("Y", 12)),
    ("Roaring Traders", ("U", 20))
],
    "reapers": [
        ("Reapers Hideout", ("I", 12))
    ],
    "Middle": [
        ("Middle of the map", ("M", 13))
    ], 
"Siren Shrines": [
    ("Shrine of the Coral Tomb", ("H", 5)),
    ("Shrine of Ocean's Fortune", ("D", 14)),
    ("Shrine of Ancient Tears", ("N", 20)),
    ("Shrine of Tribute", ("G", 18)),
    ("Shrine of Hungering", ("Q", 5)),
    ("Shrine of Flooded Embrace", ("N", 12))
],
"Siren Treasuries": [
    ("Treasury of Sunken Shores", ("D", 3)),
    ("Treasury of the Lost Ancients", ("H", 15)),
    ("Treasury of the Secret Wilds", ("L", 3))
], 
}


# Helper functions
def convert_coordinates_to_location(x, y):
    letter = chr(int(x) + ord('A') - 1)
    number = str(int(y))
    return (letter, number)


def calculate_speed(ship_direction, wind_direction):
    angle_difference = abs((ship_direction - wind_direction + 360) % 360)
    if angle_difference < 45:
        speed = shipclasses.loc[SHIP - 1, 'full_speed']
    elif angle_difference < 90:
        speed = shipclasses.loc[SHIP - 1, 'partial_speed']
    else:
        speed = shipclasses.loc[SHIP - 1, 'no_wind']
    return speed


def calculate_time_to_destination(start, end, wind_direction):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)
    angle_to_destination = math.degrees(math.atan2(dy, dx)) % 360

    speed = calculate_speed(angle_to_destination, wind_direction)
    return (distance * speed) / 60  # Convert to minutes


def calculate_cardinal_direction(angle):
    cardinal_directions = [
        ('E', 348.75, 11.25), ('SSE', 11.25, 33.75), ('SE', 33.75, 56.25),
        ('ESE', 56.25, 78.75), ('S', 78.75, 101.25), ('SWS', 101.25, 123.75),
        ('SW', 123.75, 146.25), ('WSW', 146.25, 168.75), ('W', 168.75, 191.25),
        ('WNW', 191.25, 213.75), ('NW', 213.75, 236.25), ('NNW', 236.25, 258.75),
        ('N', 258.75, 281.25), ('NNE', 281.25, 303.75), ('NE', 303.75, 326.25),
        ('ENE', 326.25, 348.75)
    ]
    for direction, lower, upper in cardinal_directions:
        if lower <= abs(angle) < upper:
            return direction
    return 'Unknown'


def find_fastest_turning_route(start, outposts, wind_direction):
    def calculate_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_distance_to_line(point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denom = calculate_distance(line_start, line_end)
        return num / denom if denom != 0 else float('inf')

    fastest_turning_routes = []
    for outpost_name, outpost_location in outposts:
        # Calculate straight-line time
        straight_line_time = calculate_time_to_destination(start, outpost_location, wind_direction)
        angle_to_outpost = math.degrees(math.atan2(outpost_location[1] - start[1], outpost_location[0] - start[0])) % 360

        direction_to_outpost = calculate_cardinal_direction(angle_to_outpost)
        end_location_name = convert_coordinates_to_location(outpost_location[0], outpost_location[1])

        # Generate potential turn locations in a 26x26 grid around the start
        potential_turns = []
        grid_size = 26
        for dx in range(-grid_size // 2, grid_size // 2 + 1):
            for dy in range(-grid_size // 2, grid_size // 2 + 1):
                turn_x = start[0] + dx
                turn_y = start[1] + dy

                # Check if the turn location is within 2 units of the start location
                if calculate_distance((turn_x, turn_y), start) < 2:
                    continue

                angle_to_turn = math.degrees(math.atan2(dy, dx)) % 360

                turn_direction = calculate_cardinal_direction(angle_to_turn)
                turn_location_name = convert_coordinates_to_location(turn_x, turn_y)
                potential_turns.append((turn_x, turn_y, turn_direction, turn_location_name))

        fastest_time = float('inf')
        fastest_turn = None
        fastest_turn_direction = None
        fastest_turn_original = None
        direction_pre_turn = None
        direction_post_turn = None

        for (turn_x, turn_y, turn_direction, turn_location_name) in potential_turns:
            turn_location = (turn_x, turn_y)
            time_to_turn = calculate_time_to_destination(start, turn_location, wind_direction)
            time_from_turn_to_outpost = calculate_time_to_destination(turn_location, outpost_location, wind_direction)
            total_time = time_to_turn + time_from_turn_to_outpost

            # Ensure the turn is not within 1 unit of the straight-line path
            distance_to_straight_line = calculate_distance_to_line(turn_location, start, outpost_location)
            if total_time < fastest_time and distance_to_straight_line >= 1:
                fastest_time = total_time
                fastest_turn_original = turn_location
                fastest_turn = turn_location_name
                fastest_turn_direction = turn_direction
                angle_pre_turn = math.degrees(math.atan2(turn_location[1] - start[1], turn_location[0] - start[0])) % 360

                angle_post_turn = math.degrees(math.atan2(outpost_location[1] - turn_location[1], outpost_location[0] - turn_location[0])) % 360

                direction_pre_turn = calculate_cardinal_direction(angle_pre_turn)
                direction_post_turn = calculate_cardinal_direction(angle_post_turn)

        if fastest_turn_original is None or fastest_time >= straight_line_time:
            fastest_turning_routes.append((round(straight_line_time), outpost_name, direction_to_outpost, end_location_name, None, None, None))

        else:
            fastest_turning_routes.append((round(fastest_time), outpost_name, direction_pre_turn, end_location_name, fastest_turn_original, fastest_turn, direction_post_turn))

    fastest_turning_routes.sort(key=lambda x: x[0])  # Sort by the total travel time
    return fastest_turning_routes


def find_fastest_turning_route(start, outposts, wind_direction):
    def calculate_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_distance_to_line(point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denom = calculate_distance(line_start, line_end)
        return num / denom if denom != 0 else float('inf')

    fastest_turning_routes = []
    for outpost_name, outpost_location in outposts:
        # Calculate straight-line time
        straight_line_time = calculate_time_to_destination(start, outpost_location, wind_direction)
        angle_to_outpost = math.degrees(math.atan2(outpost_location[1] - start[1], outpost_location[0] - start[0])) % 360

        direction_to_outpost = calculate_cardinal_direction(angle_to_outpost)
        end_location_name = convert_coordinates_to_location(outpost_location[0], outpost_location[1])

        # Generate potential turn locations in a 26x26 grid around the start
        potential_turns = []
        grid_size = 26
        for dx in range(-grid_size // 2, grid_size // 2 + 1):
            for dy in range(-grid_size // 2, grid_size // 2 + 1):
                turn_x = start[0] + dx
                turn_y = start[1] + dy

                # Check if the turn location is within 2 units of the start location
                if calculate_distance((turn_x, turn_y), start) < 2:
                    continue

                angle_to_turn = math.degrees(math.atan2(dy, dx)) % 360

                turn_direction = calculate_cardinal_direction(angle_to_turn)
                turn_location_name = convert_coordinates_to_location(turn_x, turn_y)
                potential_turns.append((turn_x, turn_y, turn_direction, turn_location_name))

        fastest_time = float('inf')
        fastest_turn = None
        fastest_turn_direction = None
        fastest_turn_original = None
        direction_pre_turn = None
        direction_post_turn = None

        for (turn_x, turn_y, turn_direction, turn_location_name) in potential_turns:
            turn_location = (turn_x, turn_y)
            time_to_turn = calculate_time_to_destination(start, turn_location, wind_direction)
            time_from_turn_to_outpost = calculate_time_to_destination(turn_location, outpost_location, wind_direction)
            total_time = time_to_turn + time_from_turn_to_outpost

            # Ensure the turn is not within 1 unit of the straight-line path
            distance_to_straight_line = calculate_distance_to_line(turn_location, start, outpost_location)
            if total_time < fastest_time and distance_to_straight_line >= 1:
                fastest_time = total_time
                fastest_turn_original = turn_location
                fastest_turn = turn_location_name
                fastest_turn_direction = turn_direction
                angle_pre_turn = math.degrees(math.atan2(turn_location[1] - start[1], turn_location[0] - start[0])) % 360

                angle_post_turn = math.degrees(math.atan2(outpost_location[1] - turn_location[1], outpost_location[0] - turn_location[0])) % 360

                direction_pre_turn = calculate_cardinal_direction(angle_pre_turn)
                direction_post_turn = calculate_cardinal_direction(angle_post_turn)

        if fastest_turn_original is None or fastest_time >= straight_line_time:
            fastest_turning_routes.append((round(straight_line_time), outpost_name, direction_to_outpost, end_location_name, None, None, None))

        else:
            fastest_turning_routes.append((round(fastest_time), outpost_name, direction_pre_turn, end_location_name, fastest_turn_original, fastest_turn, direction_post_turn))

    fastest_turning_routes.sort(key=lambda x: x[0])  # Sort by the total travel time
    return fastest_turning_routes

def load_image_from_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            img = Image.open(io.BytesIO(response.read()))
            return np.array(img)
    except Exception as e:
        st.error(f"Error loading the background image: {e}")
        return None

def plot_selected_routes_with_background(routes, background_image_url, destination_type, title_prefix="Fastest route to"):
    background = load_image_from_url(background_image_url)
    if background is None:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.imshow(background, extent=[1, 27, 1, 27])  # Set the extent of the image
    ax.plot(start[0], start[1], 'bo')
    ax.text(start[0], start[1], ' Start', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    ax.set_xticks(range(1, 27))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(26)], fontsize=10)  # Label x-axis with letters A-Z
    ax.set_yticks(range(1, 27))
    ax.set_yticklabels(range(1, 27), fontsize=10)

    ax.grid(True)

    fastest_route = min(routes, key=lambda route: route[0]) if routes else None

    for route in routes:
        time_to_outpost, outpost_name, direction_pre_turn, end_location_name, fastest_turn_original, fastest_turn, fastest_turn_direction = route
        end_loc = (ord(end_location_name[0]) - ord('A') + 1, int(end_location_name[1]))

        if fastest_turn_original:
            # Plot the turning route
            turn_x, turn_y = fastest_turn_original
            
            marker = "g*" if route == fastest_route else "ko"
            markersize = 20 if route == fastest_route else 10
            markerwidth = 25 if route == fastest_route else 12
            

            ax.plot(end_loc[0], end_loc[1], marker, markersize = markersize, linewidth = markerwidth)
            ax.text(end_loc[0], end_loc[1], f' {outpost_name} ({end_location_name})', fontsize=12, verticalalignment='top', horizontalalignment='left')
            line_style_turn = 'g--' if route == fastest_route else 'k--'
            line_style_final = 'g-' if route == fastest_route else 'k-'
            linewidth_turn = 5 if route == fastest_route else 1
            linewidth_final = 5 if route == fastest_route else 1
            color_turn_point = 'go' if route == fastest_route else 'ko'

            ax.plot([start[0], turn_x], [start[1], turn_y], line_style_turn, linewidth=linewidth_turn)
            ax.plot([turn_x, end_loc[0]], [turn_y, end_loc[1]], line_style_final, linewidth=linewidth_final)
            ax.plot(turn_x, turn_y, color_turn_point)
            ax.text(turn_x, turn_y, f' Turn ({fastest_turn})', fontsize=10, verticalalignment='bottom', horizontalalignment='right')
        else:
            # Plot the straight-line route
            marker = "g*" if route == fastest_route else "ko"
            markersize = 20 if route == fastest_route else 9
            markerwidth = 25 if route == fastest_route else 12
            ax.plot(end_loc[0], end_loc[1], marker, markersize = markersize, linewidth = markerwidth)
            ax.text(end_loc[0], end_loc[1], f' {outpost_name} ({end_location_name})', fontsize=12, verticalalignment='top', horizontalalignment='left')
            line_style_final = 'g-' if route == fastest_route else 'k-'
            linewidth_final = 5 if route == fastest_route else 1
            ax.plot([start[0], end_loc[0]], [start[1], end_loc[1]], line_style_final, linewidth=linewidth_final)

    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.set_title(f"{title_prefix} {destination_type}", fontsize=16)
    ax.invert_yaxis()  # Invert only the grid's y-axis, not the background image
    st.pyplot(plt.gcf())


# Streamlit Interface
st.title("Ship Routing Finder")

# User Inputs
X = st.selectbox("Start X Coordinate (Letter)", [chr(ord('A') + i) for i in range(26)])
Y = st.selectbox("Start Y Coordinate (Number)", list(range(1, 27)))
start = (ord(X.upper()) - ord('A') + 1, Y)

S = st.selectbox("Ship Type", ["S", "B", "G"])
W = st.selectbox("Wind Direction", list(wind_directions.keys()))
D = st.selectbox("Destination Type", list(Destinations.keys()))

# Ship selection
SHIP = Chosen_SHIP[S]

# Wind Direction
wind_direction = wind_directions[W]

if D not in Destinations:
    st.error("Invalid destination type. Please choose a valid destination type.")
else:
    destinations = Destinations[D]
    converted_destinations = [(name, (ord(loc[0]) - ord('A') + 1, loc[1])) for name, loc in destinations]

    # Find the fastest route
    fastest_turning_routes = find_fastest_turning_route(start, converted_destinations, wind_direction)
    FR = fastest_turning_routes[0]

    # Display the fastest route information
    if FR[6] is None:
        st.write(f"Fastest Route is a straight line heading {FR[2]} to reach {FR[1]}")
        st.write(f"Fastest route will take about {FR[0]} minutes")
    else:
        st.write(f"Route will take about {FR[0]} minutes to reach {FR[1]}")
        st.write(f"Head {FR[2]}, turn at {FR[3]} after turning head {FR[6]}")

    # Plot with background
background_image_url = 'https://raw.githubusercontent.com/Worryingcow/SOT_GPS/main/images/MAP_NN_NB.png?raw=true'
st.write("### Fastest Routes Plot with Background")
plot_selected_routes_with_background(fastest_turning_routes, background_image_url, D)
