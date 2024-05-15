#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:01:25 2024

@author: WorryingCow

"""


import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import io
from urllib.parse import quote


#Im updating this to frequently where I dont want to delete and publish a new streamlit app every time I just copy paste new code, 
#This is V6.3 
#Major addtions are sliders, stops, and something else I cant remember


#Stable version with choice
#Final Destination is removed from stop choice
#I would like to optimize the route, so you can put final destination and all the places you want to go and it tells you the fastest way
#IE order of stops doesnt matter


# Ship Speed DataFrame
shipclasses = pd.DataFrame({
    'class': ['Sloop', 'Brig', 'Gally'],
    'full_speed': [40, 35, 35],  # Speeds in seconds/unit
    'partial_speed': [45, 40, 40], #dummy speeds but close for brig and gal
    'no_wind': [50, 55, 90]
})

# Ship Type Mapping
Chosen_SHIP = {
    "Sloop": 1,
    "Briggiantine": 2,  # Corrected spelling
    "Galleon": 3
}

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
        ("Port Merrick", ("D", 10))
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
    ]
}

Islands = {
    "Ancient Spires Outpost": [("Ancient Spires Outpost", ("Q", 17))],
    "Dagger Tooth Outpost": [("Dagger Tooth Outpost", ("M", 8))],
    "Galleon's Grave Outpost": [("Galleon's Grave Outpost", ("R", 8))],
    "Golden Sands Outpost": [("Golden Sands Outpost", ("D", 10))],
    "Morrow's Peak Outpost": [("Morrow's Peak Outpost", ("V", 17))],
    "Plunder Outpost": [("Plunder Outpost", ("J", 18))],
    "Sanctuary Outpost": [("Sanctuary Outpost", ("F", 7))],
    "Brian's Bazaar": [("Brian's Bazaar", ("Y", 12))],
    "Roaring Traders": [("Roaring Traders", ("U", 20))],
    "Stephen's Spoils": [("Stephen's Spoils", ("L", 15))],
    "The Finest Trading Post": [("The Finest Trading Post", ("F", 17))],
    "The North Star Seapost": [("The North Star Seapost", ("H", 10))],
    "The Spoils of Plenty Store": [("The Spoils of Plenty Store", ("B", 7))],
    "The Wild Treasures Store": [("The Wild Treasures Store", ("O", 4))],
    "Three Paces East Seapost": [("Three Paces East Seapost", ("S", 9))],
    "Barnacle Cay": [("Barnacle Cay", ("O", 15))],
    "Black Sand Atoll": [("Black Sand Atoll", ("O", 3))],
    "Black Water Enclave": [("Black Water Enclave", ("R", 5))],
    "Blind Man's Lagoon": [("Blind Man's Lagoon", ("N", 6))],
    "Booty Isle": [("Booty Isle", ("K", 20))],
    "Boulder Cay": [("Boulder Cay", ("G", 5))],
    "Brimstone Rock": [("Brimstone Rock", ("X", 18))],
    "Castaway Isle": [("Castaway Isle", ("K", 14))],
    "Chicken Isle": [("Chicken Isle", ("I", 16))],
    "Cinder Islet": [("Cinder Islet", ("U", 14))],
    "Cursewater Shores": [("Cursewater Shores", ("Y", 13))],
    "Cutlass Cay": [("Cutlass Cay", ("M", 18))],
    "Flame's End": [("Flame's End", ("V", 19))],
    "Fools Lagoon": [("Fools Lagoon", ("I", 14))],
    "Glowstone Cay": [("Glowstone Cay", ("Z", 18))],
    "Isle of Last Words": [("Isle of Last Words", ("O", 9))],
    "Lagoon of Whispers": [("Lagoon of Whispers", ("D", 12))],
    "Liar's Backbone": [("Liar's Backbone", ("S", 11))],
    "Lonely Isle": [("Lonely Isle", ("G", 8))],
    "Lookout Point": [("Lookout Point", ("I", 20))],
    "Magma's Tide": [("Magma's Tide", ("Y", 20))],
    "Mutineer Rock": [("Mutineer Rock", ("N", 19))],
    "Old Salts Atoll": [("Old Salts Atoll", ("F", 18))],
    "Paradise Spring": [("Paradise Spring", ("L", 17))],
    "Picaroon Palms": [("Picaroon Palms", ("I", 4))],
    "Plunderer's Plight": [("Plunderer's Plight", ("Q", 6))],
    "Port Merrick": [ ("Port Merrick", ("D", 10))],
    "Rapier Cay": [("Rapier Cay", ("D", 8))],
    "Roaring Sands": [("Roaring Sands", ("U", 21))],
    "Rum Runner Isle": [("Rum Runner Isle", ("H", 9))],
    "Salty Sands": [("Salty Sands", ("G", 3))],
    "Sandy Shallows": [("Sandy Shallows", ("D", 5))],
    "Schored Pass": [("Schored Pass", ("X", 11))],
    "Scurvy Isley": [("Scurvy Isley", ("K", 4))],
    "Sea Dog's Rest": [("Sea Dog's Rest", ("C", 11))],
    "Shark Tooth Key": [("Shark Tooth Key", ("P", 13))],
    "Shiver Retreat": [("Shiver Retreat", ("Q", 11))],
    "The Forsaken Brink": [("The Forsaken Brink", ("U", 16))],
    "Tribute Peak": [("Tribute Peak", ("Y", 2))],
    "Tri-Rock Isle": [("Tri-Rock Isle", ("R", 10))],
    "Twin Groves": [("Twin Groves", ("H", 11))],
    "Ashen Reaches": [("Ashen Reaches", ("V", 23))],
    "Cannon Cove": [("Cannon Cove", ("G", 10))],
    "Crescent Isle": [("Crescent Isle", ("B", 9))],
    "Crook's Hollow": [("Crook's Hollow", ("M", 16))],
    "Devil's Ridge": [("Devil's Ridge", ("P", 19))],
    "Discovery Ridge": [("Discovery Ridge", ("E", 17))],
    "Fetcher's Rest": [("Fetcher's Rest", ("V", 12))],
    "Flintlock Peninsula": [("Flintlock Peninsula", ("W", 14))],
    "Kraken's Fall": [("Kraken's Fall", ("R", 12))],
    "Lone Cove": [("Lone Cove", ("H", 6))],
    "Marauder's Arch": [("Marauder's Arch", ("Q", 3))],
    "Mermaid's Hideaway": [("Mermaid's Hideaway", ("B", 13))],
    "Old Faithful Isle": [("Old Faithful Isle", ("M", 4))],
    "Plunder Valley": [("Plunder Valley", ("G", 16))],
    "Ruby's Fall": [("Ruby's Fall", ("Y", 16))],
    "Sailor's Bounty": [("Sailor's Bounty", ("C", 4))],
    "Shark Bait Cove": [("Shark Bait Cove", ("H", 19))],
    "Shipwreck Bay": [("Shipwreck Bay", ("M", 10))],
    "Smugglers' Bay": [("Smugglers' Bay", ("F", 3))],
    "Snake Island": [("Snake Island", ("K", 16))],
    "The Crooked Masts": [("The Crooked Masts", ("O", 11))],
    "The Devil's Thirst": [("The Devil's Thirst", ("W", 21))],
    "The Sunken Grove": [("The Sunken Grove", ("P", 7))],
    "Thieves' Haven": [("Thieves' Haven", ("L", 20))],
    "Wanderers Refuge": [("Wanderers Refuge", ("F", 12))],
    "Hidden Spring Keep": [("Hidden Spring Keep", ("I", 8))],
    "Keel Haul Fort": [("Keel Haul Fort", ("C", 6))],
    "Kraken Watchtower": [("Kraken Watchtower", ("L", 6))],
    "Lost Gold Fort": [("Lost Gold Fort", ("H", 17))],
    "Molten Sands Fortress": [("Molten Sands Fortress", ("Z", 11))],
    "Old Boot Fort": [("Old Boot Fort", ("L", 14))],
    "Sailor's Knot Stronghold": [("Sailor's Knot Stronghold", ("E", 14))],
    "Shark Fin Camp": [("Shark Fin Camp", ("P", 5))],
    "Skull Keep": [("Skull Keep", ("P", 9))],
    "The Crow's Nest Fortress": [("The Crow's Nest Fortress", ("O", 17))]
}
 #Every Island and their coordinates

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

@st.cache_data
def load_image_from_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            img = Image.open(io.BytesIO(response.read()))
            return np.array(img)
    except Exception as e:
        st.error(f"Error loading the background image: {e}")
        return None



# Function to plot routes with background including stops
def plot_selected_routes_with_background(routes, background_image_url, stops, destination_type, title_prefix="Fastest route to"):
    # Load the background image from the URL
    background = load_image_from_url(background_image_url)
    if background is None:
        return



    # Create a figure and axis with a fixed aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Display the background image
    ax.imshow(background, extent=[1, 27, 1, 27])

    # Plot the starting point with a blue dot and label
    ax.plot(start[0], start[1], 'bo')
    ax.text(start[0], start[1], ' Start', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Set x and y ticks to represent the grid
    ax.set_xticks(range(1, 27))
    ax.set_xticklabels([chr(ord('A') + i) for i in range(26)], fontsize=10)  # Label x-axis with letters A-Z
    ax.set_yticks(range(1, 27))
    ax.set_yticklabels(range(1, 27), fontsize=10)
    ax.grid(True)

    # Plot stops with red dots and labels
    for stop_name, stop_coord in stops:
        stop_x, stop_y = stop_coord
        ax.plot(stop_x, stop_y, 'ro')  # Red dot for stops
        ax.text(stop_x, stop_y, f' {stop_name} ({chr(ord("A") + stop_x - 1)}, {stop_y})', fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Determine the fastest route from the list of routes
    fastest_route = min(routes, key=lambda route: route[0]) if routes else None

    # Collect all coordinates for the fastest route
    if fastest_route:
        all_coords = [start]  # Start with the initial point
        route_island_names = []
        for route in routes:
            time_to_outpost, outpost_name, direction_pre_turn, end_location_name, fastest_turn_original, fastest_turn, fastest_turn_direction = route
            end_loc = (ord(end_location_name[0]) - ord('A') + 1, int(end_location_name[1]))

            if fastest_turn_original:
                turn_x, turn_y = fastest_turn_original
                all_coords.append((turn_x, turn_y))  # Append turning point coordinates
                route_island_names.append((f'Turn ({turn_x},{turn_y})', (turn_x, turn_y)))
            all_coords.append(end_loc)  # Append destination coordinates
            route_island_names.append((outpost_name, end_loc))

        # Extract x and y coordinates for plotting
        x_coords, y_coords = zip(*all_coords)


#Green line and dots appear on top of others if plot after
        # Plot the fastest route as a single continuous path with a green line and arrows for direction
        for i in range(len(x_coords) - 1):
            ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], 'g-', linewidth=3)
            ax.arrow(x_coords[i], y_coords[i], x_coords[i + 1] - x_coords[i], y_coords[i + 1] - y_coords[i],
                     head_width=0.5, head_length=0.5, fc='green', ec='green')

        # Label the islands on the fastest route
        for (label, (x, y)) in route_island_names:
            ax.plot(x, y, 'go', markersize=1)  # Mark each point along the fastest route
            ax.text(x, y, f' {label} ({chr(ord("A") + x - 1)}, {y})', fontsize=12, verticalalignment='top', horizontalalignment='left')


 # Mark the final destination with a red X
        final_x, final_y = x_coords[-1], y_coords[-1]
        ax.plot(final_x, final_y, 'rx', markersize=13, mew=3)  # Red X for final destination
        ax.text(final_x, final_y, f' {route_island_names[-1][0]} ({chr(ord("A") + final_x - 1)}, {final_y})', fontsize=12, verticalalignment='top', horizontalalignment='left')


    # Add wind direction arrow in the bottom right corner #Im honestly not a fan of this
  #  wind_direction_angle = wind_directions[W]
 #   wind_x = 3  # Adjust position as needed
 #   wind_y = 26  # Adjust position as needed
 #   ax.arrow(wind_x, wind_y, 2 * math.cos(math.radians(wind_direction_angle)), 2 * math.sin(math.radians(wind_direction_angle)),
 #            head_width=0.5, head_length=1, fc='black', ec='black', linewidth = 5)
  #  ax.text(wind_x + 1, wind_y- 1, f'Wind Direction: {W}', fontsize=12, color='black', verticalalignment='top', horizontalalignment='center')

    # Set axis labels and title
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.set_title(f"{title_prefix} {destination_type}", fontsize=16)

    # Invert y-axis to match the grid layout
    ax.invert_yaxis()  # Invert only the grid's y-axis, not the background image

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

    # Clear the figure after plotting
    plt.clf()


# Streamlit Interface
st.title("Ship Routing Finder")

X_Coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Toggle between Destination and Island Categories
destination_type = st.radio("Choose your target type:", ("Island Categories", "Specific Island"))

# User Inputs
X = st.select_slider("Start X Coordinate (Letter)", [chr(ord('A') + i) for i in range(26)])
Y = st.select_slider("Start Y Coordinate (Number)", list(range(1, 27)))
start = (ord(X.upper()) - ord('A') + 1, Y)

S = st.select_slider("Ship Type", ["Sloop", "Briggiantine", "Galleon"])
W = st.select_slider("Wind Direction", list(wind_directions.keys()))

# Ensure the final destination is appended correctly
if destination_type == "Island Categories":
    if 'Destinations' in globals() and Destinations:  # Check if 'Destinations' is defined and not empty
        D = st.selectbox("Destination Categories", list(Destinations.keys()))
        final_destination = (ord(Destinations[D][0][1][0]) - ord('A') + 1, Destinations[D][0][1][1])
        final_destination_name = Destinations[D][0][0]
    else:
        st.error("Destination data is not available.")
else:
    if 'Islands' in globals() and Islands:  # Check if 'Islands' is defined and not empty
        D = st.selectbox("Specific Islands", list(Islands.keys()))
        final_destination = (ord(Islands[D][0][1][0]) - ord('A') + 1, Islands[D][0][1][1])
        final_destination_name = Islands[D][0][0]
    else:
        st.error("Island data is not available.")

# Remove the final destination from the list of stops
filtered_destinations = {key: val for key, val in Destinations.items() if key != D}
filtered_islands = {key: val for key, val in Islands.items() if key != D}

# Add Stops
num_stops = st.number_input("Number of Stops", min_value=0, max_value=5, value=0)
stops = []
stop_names = []
for i in range(num_stops):
    stop_type = st.radio(f"Stop {i+1} Type", ("Island Categories", "Specific Island"), key=f"stop_type_{i}")
    if stop_type == "Island Categories":
        stop_category = st.selectbox(f"Stop {i+1} Category", list(filtered_destinations.keys()), key=f"stop_category_{i}")
        stops.append((ord(filtered_destinations[stop_category][0][1][0]) - ord('A') + 1, filtered_destinations[stop_category][0][1][1]))
        stop_names.append(filtered_destinations[stop_category][0][0])
    else:
        stop_island = st.selectbox(f"Stop {i+1} Island", list(filtered_islands.keys()), key=f"stop_island_{i}")
        stops.append((ord(filtered_islands[stop_island][0][1][0]) - ord('A') + 1, filtered_islands[stop_island][0][1][1]))
        stop_names.append(filtered_islands[stop_island][0][0])

route_points = [start] + stops + [final_destination]
destination_names = ["Start"] + stop_names + [final_destination_name]

# Function to find the fastest route with stops
def find_fastest_route_with_stops(route_points, destination_names, wind_direction):
    total_time = 0
    detailed_route = []
    
    for i in range(len(route_points) - 1):
        start_point = route_points[i]
        end_point = route_points[i + 1]
        end_name = destination_names[i + 1]
        
        # Find the fastest route between current point and next point
        destinations = [(end_name, end_point)]
        fastest_turning_routes = find_fastest_turning_route(start_point, destinations, wind_direction)
        fastest_route = fastest_turning_routes[0]
        
        total_time += fastest_route[0]
        detailed_route.append((fastest_route[0], end_name, fastest_route[2], fastest_route[3], fastest_route[4], fastest_route[5], fastest_route[6]))
    
    return total_time, detailed_route

# Find the fastest route with stops
SHIP = Chosen_SHIP[S]
wind_direction = wind_directions[W]
total_time_with_stops, detailed_route_with_stops = find_fastest_route_with_stops(route_points, destination_names, wind_direction)

# Calculate the travel time without stops for comparison
route_points_without_stops = [start, final_destination]
destination_names_without_stops = ["Start", final_destination_name]
total_time_without_stops, _ = find_fastest_route_with_stops(route_points_without_stops, destination_names_without_stops, wind_direction)

# Display the total route information
st.write(f"Total travel time is about {total_time_with_stops} minutes")

for i, route in enumerate(detailed_route_with_stops):
    if route[6] is None:
        st.write(f" Head {route[2]} to reach {route[1]}")
    else:
        st.write(f"leg {i+1}  will take about {route[0]} minutes to reach {route[1]}")
        st.write(f"Head {route[2]}, turn at {route[3]} after turning head {route[6]}")

# Calculate and display the additional time added by each stop
if num_stops > 0:
    additional_time = total_time_with_stops - total_time_without_stops
    st.write(f"Stop adds about {additional_time} minutes")

# Plot with background
background_image_url = 'https://raw.githubusercontent.com/Worryingcow/SOT_GPS/6002bcfeac691b3611cf3fed59c16ae84215b0cf/images/MAP_NN_NB.png'
st.write("### Fastest Routes Plot with Background")
plot_selected_routes_with_background(detailed_route_with_stops, background_image_url, list(zip(stop_names, stops)), D)


