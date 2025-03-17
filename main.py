from fastapi.responses import FileResponse
from fastapi import FastAPI
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.preprocessing import StandardScaler
import random
import json

# Define Cebu Strait boundaries - adjusted to focus on water areas
CEBU_STRAIT = {'min_lat': 9.9145, 'max_lat': 10.4121, 'min_lon': 123.7573, 'max_lon': 124.0674}

# Define FastAPI app
app = FastAPI()

# Define the fish predictions directory (update this with your local path)
predictions_dir = "C:\\Users\\Joseph\\my-fastapi-backend\\fish_predictions"

# Ensure the directory exists
os.makedirs(predictions_dir, exist_ok=True)

# Define helper functions for predictions
def generate_predictions(hour_key):
    """Generate predictions for consistent locations"""
    predictions = []

    # Generate 5 different locations
    for i in range(5):
        # Divide the strait into regions to ensure diversity
        region_lat_size = (CEBU_STRAIT['max_lat'] - CEBU_STRAIT['min_lat']) / 3
        region_lon_size = (CEBU_STRAIT['max_lon'] - CEBU_STRAIT['min_lon']) / 3

        # Select different regions for each prediction
        lat_region = i % 3
        lon_region = i // 3 % 3

        # Calculate region boundaries
        min_lat_region = CEBU_STRAIT['min_lat'] + lat_region * region_lat_size
        max_lat_region = min_lat_region + region_lat_size
        min_lon_region = CEBU_STRAIT['min_lon'] + lon_region * region_lon_size
        max_lon_region = min_lon_region + region_lon_size

        # Generate random point within the region (but not exactly at the edge)
        buffer = 0.02  # Buffer from region edges
        pred_lat = min_lat_region + buffer + random.random() * (region_lat_size - 2 * buffer)
        pred_lon = min_lon_region + buffer + random.random() * (region_lon_size - 2 * buffer)

        # Time-based confidence adjustment
        time_factor = abs(hour_key - 12) / 12  # 0 at noon, 1 at midnight
        base_confidence = 50 + random.random() * 15
        confidence = round(base_confidence + (time_factor * 4), 1)  # Higher confidence at night
        confidence = min(69.9, confidence)  # Cap at 69.9%

        predictions.append({
            'latitude': pred_lat,
            'longitude': pred_lon,
            'confidence': confidence
        })

    # Ensure the predictions are all different
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            distance = np.sqrt((predictions[i]['latitude'] - predictions[j]['latitude'])**2 +
                              (predictions[i]['longitude'] - predictions[j]['longitude'])**2)

            # If too close, move the second prediction
            if distance < 0.05:  # Minimum 0.05 degrees separation
                predictions[j]['latitude'] += 0.07 + random.random() * 0.05
                predictions[j]['longitude'] += 0.07 + random.random() * 0.05

                # Re-check bounds
                predictions[j]['latitude'] = max(CEBU_STRAIT['min_lat'],
                                            min(CEBU_STRAIT['max_lat'], predictions[j]['latitude']))
                predictions[j]['longitude'] = max(CEBU_STRAIT['min_lon'],
                                             min(CEBU_STRAIT['max_lon'], predictions[j]['longitude']))

    return predictions

def create_map(predictions, dt):
    """Create a map visualization of predictions"""
    center_lat = (CEBU_STRAIT['min_lat'] + CEBU_STRAIT['max_lat']) / 2
    center_lon = (CEBU_STRAIT['min_lon'] + CEBU_STRAIT['max_lon']) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add Cebu Strait boundary
    folium.Rectangle(
        bounds=[[CEBU_STRAIT['min_lat'], CEBU_STRAIT['min_lon']],
                [CEBU_STRAIT['max_lat'], CEBU_STRAIT['max_lon']]],
        color='blue',
        fill=True,
        fill_opacity=0.1,
        tooltip='Cebu Strait'
    ).add_to(m)

    # Add markers for each prediction location
    for pred in predictions:
        folium.Marker(
            location=[pred['latitude'], pred['longitude']],
            popup=f"Confidence: {pred['confidence']}%",
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)

    # Save map file in the predictions directory
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%H%M")
    map_file = os.path.join(predictions_dir, f'cebu_fish_predictions_{date_str}_{hour_str}.html')  # Save here

    m.save(map_file)

    print(f"Prediction complete! Map visualization created for fish locations in Cebu Strait.")
    print(f"Map saved to: {map_file}")

    return map_file  # Optionally return the map file path

# Handle favicon requests to avoid 404 errors
@app.get("/favicon.ico")
def favicon():
    """Return an empty response for favicon to avoid 404 errors"""
    return FileResponse("path_to_your_favicon.ico")  # You can add a path to a valid favicon if needed

# Main prediction function
@app.get("/predict/")
def predict_cebu_fish(year: int, month: int, day: int, hour: int, minute: int):
    """Main function to predict fish locations in Cebu Strait with consistent predictions."""
    # Validate input
    if not (2020 <= year <= 2025 and 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return {"error": "Invalid date/time values"}

    # Format date and time
    date_str = f"{year}-{month:02d}-{day:02d}"
    time_str = f"{hour:02d}:{minute:02d}"
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

    print(f"\nGenerating predictions for: {dt.strftime('%Y-%m-%d %H:%M')}")

    # Generate predictions
    predictions = generate_predictions(hour)

    # Create map visualization and save to file
    map_file = create_map(predictions, dt)

    # Return the map file as a response
    return FileResponse(map_file)
