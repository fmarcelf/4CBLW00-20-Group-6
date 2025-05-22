import pandas as pd
import folium
import json
import streamlit as st
from streamlit_folium import st_folium

# TODO: Add caching
# TODO: Clean code, add comments
# TODO: Improve performance - map reloads every user input
#       Try using session states for storing whole map on client side
# TODO: Try database instead of dataframe

st.set_page_config(layout="wide")

# Data handling, methods

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

@st.cache_data
def create_base_map():
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    # TODO: Add more functionalities
    return m

# Load data

df, data_gjs = load_data()

# Create filters
years = []
for i in range (2010, 2026):
    years.append(i)

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

lsoa = sorted(df['LSOA code'].unique())


# Visual - Page customization

st.title(" :gb: London Burglary Explorer")

# Create sidebar for user input

#TODO: Change header
st.sidebar.header("Parameters")
selected_year = st.sidebar.selectbox("Year", years)
selected_month = st.sidebar.selectbox("Month", months)
selected_lsoa = st.sidebar.selectbox("LSOA code", lsoa)

# Filter data
filtered = filter_data(df, selected_year, selected_month)

# Create test map on the right

m = create_base_map()

folium.Choropleth(
        geo_data=data_gjs,
        data=filtered,
        columns=['LSOA code', 'Burglaries'],
        key_on='feature.properties.LSOA21CD',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Nr of burglaries"
        ).add_to(m)


# Display map
st_folium(m, width=2000)

# TODO: Display additional LSOA statistics

st.markdown(
        "<h2 style='text-align:center;'> Statistics for Your LSOA</h2>",
        unsafe_allow_html=True
        )



