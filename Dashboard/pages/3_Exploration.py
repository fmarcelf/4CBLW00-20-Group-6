import streamlit as st
import copy
import json
import pydeck as pdk
import pandas as pd
import numpy as np  # Needed for percentile and clipping
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploration Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Data/data.csv")
    with open("Data/london.geojson", "r") as f:
        geojson_data = json.load(f)
    return df, geojson_data

@st.cache_data
def filter_data(df, year_range, month_range):
    filtered = df[
        (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1]) &
        (df["Month"] >= month_range[0]) & (df["Month"] <= month_range[1])
    ]
    return filtered

def add_burglaries_to_geojson(geojson, df):
    burglary_map = df.groupby('LSOA code')['Burglaries'].sum().to_dict()
    geojson_copy = copy.deepcopy(geojson)

    burglary_values = np.array(list(burglary_map.values()))

    if len(burglary_values) == 0:
        lower = upper = min_b = max_b = 0
    else:
        lower = np.percentile(burglary_values, 5)
        upper = np.percentile(burglary_values, 95)
        burglary_values_clipped = np.clip(burglary_values, lower, upper)
        min_b = burglary_values_clipped.min()
        max_b = burglary_values_clipped.max()

    for feature in geojson_copy['features']:
        lsoa_code = feature['properties']['LSOA21CD']
        raw_count = burglary_map.get(lsoa_code, 0)
        clipped_count = np.clip(raw_count, lower, upper) if max_b != min_b else 0
        normalized = (clipped_count - min_b) / (max_b - min_b) if max_b != min_b else 0
        feature['properties']['burglaries'] = int(raw_count)
        feature['properties']['normalized'] = normalized

    return geojson_copy

# Load data
df, geojson_data = load_data()

years = list(range(2010, 2026))
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

lsoa = sorted(df['LSOA code'].unique())

# Define filter options with the requested order and labels
filter_options = {
    "All Time (2010–2025)": "all",
    "COVID-19 Period (March 2020 – March 2021)": "covid",
    "Custom Range": "custom",
    "Single Year": "single",
    "Only Autumns (September-November)": "autumn",
    "Only Winters (December-February)": "winter",
    "Only Springs (March-May)": "spring",
    "Only Summers (June-August)": "summer",
}

filter_choice = st.selectbox("Select Data Filter", list(filter_options.keys()))

if filter_options[filter_choice] == "all":
    # All time and all months
    start_year, end_year = 2010, 2025
    start_month, end_month = 1, 12
    show_sliders = False

elif filter_options[filter_choice] == "covid":
    # Fixed covid period: March 2020 to March 2021
    start_year, start_month = 2020, 3
    end_year, end_month = 2021, 3
    show_sliders = False

elif filter_options[filter_choice] == "custom":
    show_sliders = True
    st.markdown("#### Select Year Range")
    year_range = st.select_slider(
        "",
        options=years,
        value=(2010, 2025),
        format_func=lambda x: str(x)
    )

    st.markdown("#### Select Month Range")
    month_range = st.select_slider(
        "",
        options=months,
        value=(months[0], months[-1])
    )

    start_year, end_year = year_range
    start_month_name, end_month_name = month_range
    start_month = months.index(start_month_name) + 1
    end_month = months.index(end_month_name) + 1

elif filter_options[filter_choice] == "single":
    show_sliders = False
    single_year = st.number_input("Enter Year", min_value=2010, max_value=2025, value=2010, step=1)
    start_year = end_year = single_year
    start_month, end_month = 1, 12

else:
    # Seasons handling
    show_sliders = False
    season = filter_options[filter_choice]
    start_year, end_year = 2010, 2025

    if season == "autumn":
        # September (9) to November (11)
        start_month, end_month = 9, 11
    elif season == "winter":
        # December (12) to February (2) -- crosses year boundary, so will filter specially
        # We'll handle winter months separately after filtering data by years
        # For now assign months to cover 12,1,2 and handle later in filtering
        # So let's treat as months 12, 1, 2 but filtering will need adjustment.
        start_month, end_month = 12, 2
    elif season == "spring":
        start_month, end_month = 3, 5
    elif season == "summer":
        start_month, end_month = 6, 8
    else:
        # Fallback
        start_month, end_month = 1, 12

# Now, prepare filtered data based on chosen filter
def filter_seasonal_data(df, start_year, end_year, start_month, end_month, season=None):
    if season == "winter":
        # Winter crosses year boundary:
        # We want Dec (month=12) of previous year and Jan-Feb (month=1,2) of current year.
        # So filter years from (start_year-1) to end_year for Dec, Jan, Feb months.
        # Then filter months and years accordingly.

        # Filter rows where (Year == y and Month in [12]) OR (Year == y+1 and Month in [1,2])
        # For all years in range start_year to end_year inclusive.

        dfs = []
        for y in range(start_year, end_year + 1):
            dec_rows = df[(df['Year'] == y - 1) & (df['Month'] == 12)]
            jan_feb_rows = df[(df['Year'] == y) & (df['Month'].isin([1, 2]))]
            dfs.append(pd.concat([dec_rows, jan_feb_rows]))
        filtered = pd.concat(dfs)
        # Remove duplicates if any
        filtered = filtered.drop_duplicates()
        return filtered
    else:
        # Normal filtering for months in the same year range
        filtered = df[
            (df["Year"] >= start_year) & (df["Year"] <= end_year) &
            (df["Month"] >= start_month) & (df["Month"] <= end_month)
        ]
        return filtered

if filter_options[filter_choice] == "winter":
    filtered = filter_seasonal_data(df, start_year, end_year, start_month, end_month, season="winter")
elif filter_options[filter_choice] in ["autumn", "spring", "summer"]:
    filtered = filter_seasonal_data(df, start_year, end_year, start_month, end_month, season=filter_options[filter_choice])
else:
    filtered = filter_data(df, (start_year, end_year), (start_month, end_month))

# Validate ranges for custom and single year selections
if filter_options[filter_choice] in ["custom"]:
    if end_year < start_year:
        st.error("End Year must be greater than or equal to Start Year.")
        st.stop()

    if end_month < start_month:
        st.error("End Month must be greater than or equal to Start Month.")
        st.stop()

if filtered.empty:
    st.warning("No Data Found - Change Year and/or Month.")
else:
    gjs = add_burglaries_to_geojson(geojson_data, filtered)

    view_state = pdk.ViewState(
        latitude=51.5074,
        longitude=-0.1278,
        zoom=9,
        pitch=0
    )

    gjs_layer = pdk.Layer(
        "GeoJsonLayer",
        gjs,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="""
            [
                255,
                255 * (1 - properties.normalized),
                0
            ]
        """,
        get_line_color=[0, 0, 255],
        line_width_min_pixels=1
    )

    r = pdk.Deck(
        layers=[gjs_layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={
            "html": "<b>LSOA:</b> {LSOA21NM}<br/><b>Burglaries:</b> {burglaries}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )

    st.pydeck_chart(r)

    # For custom range, single year, or seasons, no missing combos shown since months/years are fixed or partial
    # For custom range, we can still show missing combos:
    if filter_options[filter_choice] == "custom":
        selected_years = list(range(start_year, end_year + 1))
        selected_months = list(range(start_month, end_month + 1))
        all_combinations = [(y, m) for y in selected_years for m in selected_months]
        existing_combinations = set(zip(filtered['Year'], filtered['Month']))

        missing_combinations = [
            f"{y} {months[m-1]}"
            for (y, m) in all_combinations if (y, m) not in existing_combinations
        ]

        if missing_combinations:
            st.warning(f"⚠️ No data found for these year-month combinations: {', '.join(missing_combinations)}")

# LSOA inspection

st.header("Select LSOA for inspection")
selected_lsoa = st.selectbox("LSOA code", lsoa)

lsoa_data = df[df['LSOA code'] == selected_lsoa]

nr_burglaries = lsoa_data['Burglaries'].sum()

st.markdown(
    f"<h2 style='text-align: center;'>Total Burglaries for {selected_lsoa}: <strong>{int(nr_burglaries):,}</strong></h2>",
    unsafe_allow_html=True
)

# Plot burglaries over the years
plt.figure(figsize=(16,10))
sns.lineplot(x = lsoa_data['YearMonth'], y = lsoa_data['Burglaries'])
plt.title(f"Summary of burglaries for {selected_lsoa}")
plt.xlabel("Date")
plt.ylabel("Burglary count")
plt.tight_layout()

# Show plot in Streamlit
st.pyplot(plt.gcf())
