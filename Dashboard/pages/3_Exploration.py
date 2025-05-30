import streamlit as st
import copy
import json
import pydeck as pdk
import pandas as pd

st.title("Exploration Dashboard")


@st.cache_data
def load_data():
    df = pd.read_csv("Data/data.csv")
    with open("Data/london.geojson", "r") as f:
        geojson_data = json.load(f)
    return df, geojson_data


@st.cache_data
def filter_data(df, selected_year, selected_month):
    month_num = months.index(selected_month) + 1
    filtered = df[(df['Year'] == selected_year) & (df['Month'] == month_num)]
    return filtered

def add_burglaries_to_geojson(geojson, df):
    burglary_map = df.set_index('LSOA code')['Burglaries'].to_dict()
    geojson_copy = copy.deepcopy(geojson)

    for feature in geojson_copy['features']:
        lsoa_code = feature['properties']['LSOA21CD']
        count = burglary_map.get(lsoa_code, 0)
        feature['properties']['burglaries'] = int(count)

    return geojson_copy

# Load data

df, geojson_data = load_data()

# Create filters

years = list(range(2010,2026))

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

# Create user input fields

selected_year = st.selectbox("Year", years)
selected_month = st.selectbox("Month", months)

month_num = months.index(selected_month) + 1
filtered = filter_data(df, selected_year, selected_month)

gjs = add_burglaries_to_geojson(geojson_data, filtered)

# Create Map

# Create view state for map

view_state = pdk.ViewState(
        latitude=51.5074,
        longitude=-0.1278,
        zoom=10,
        pitch=0
        )

# Create base layer
base_layer = pdk.Layer(
        "TileLayer",
        data=None,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1,
        pickable=False,
        get_tile_data=None,
        url_template = "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"
        )

# Create GeoJSON layer
#TODO: Change color filling
gjs_layer = pdk.Layer(
        "GeoJsonLayer",
        gjs,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="[255, (1 - properties.burglaries / 50) * 255, 0]",
        get_line_color=[255,255,255],
        line_width_min_pixels=1
        )

r = pdk.Deck(
        layers=[base_layer, gjs_layer],
        initial_view_state=view_state,
        tooltip=True
        )

st.pydeck_chart(r)


