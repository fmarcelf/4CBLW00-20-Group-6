# ---
# title: "testtesttest?"
# ---

import streamlit as st
import pandas as pd
import math
import datetime
import calendar
import os

# Load burglary data
df = pd.read_csv("Burglary Data - One row per LSOA CSV.csv")

# Define tasks
tasks = {
    1: {
        "name": "Drive around the assigned LSOA to maintain visibility and presence",
        "default_minutes": 5,
        "scales_with_area": True
    },
    2: {
        "name": "Engage with residents to gather tips and raise awareness",
        "default_minutes": 60,
        "scales_with_area": False
    },
    3: {
        "name": "Walk through high-risk or high-traffic zones (foot patrol)",
        "default_minutes": 1,
        "scales_with_area": True
    },
    4: {
        "name": "Visits to burglary victims or vulnerable homes",
        "default_minutes": 60,
        "scales_with_area": False
    },
    5: {
        "name": "Other: e.g., note poor lighting, check CCTV spots, write reports",
        "default_minutes": 60,
        "scales_with_area": False
    }
}

def get_upcoming_months():
    today = datetime.date.today()
    months = []
    for i in range(3):
        month_date = today.replace(day=1) + pd.DateOffset(months=i)
        months.append(calendar.month_name[month_date.month])
    return months

def calculate_Ci(row, user_tasks):
    area = row["LSOA Area Size (HA)"]
    total_minutes = 0
    for i in user_tasks:
        if user_tasks[i]["scales_with_area"]:
            total_minutes += user_tasks[i]["minutes"] * area
        else:
            total_minutes += user_tasks[i]["minutes"]
    return math.ceil(total_minutes / 120)

st.title("Ward Optimization Dashboard")

df.columns = df.columns.str.strip()

st.subheader("Choose a ward to optimize")

if "ward name" in df.columns and "ward code" in df.columns:
    df["ward_display"] = df["ward name"] + " (" + df["ward code"] + ")"
    ward_display_list = df[["ward_display", "ward name", "ward code"]].drop_duplicates()

    ward_display_list["ward_display"] = ward_display_list["ward_display"].astype(str)

    selected_display = st.selectbox("Select a ward", ["-"] + sorted(ward_display_list["ward_display"].unique()))

    if selected_display != "-":
        selected_row = ward_display_list[ward_display_list["ward_display"] == selected_display].iloc[0]
        selected_ward_name = selected_row["ward name"]
        selected_ward_code = selected_row["ward code"]
        ward_df = df[df["ward code"] == selected_ward_code].copy()
    else:
        selected_ward_name = selected_ward_code = None
        ward_df = pd.DataFrame()
else:
    st.error("Columns 'ward name' and/or 'ward code' not found in the dataset.")
    st.stop()

if "start" not in st.session_state:
    st.session_state.start = False

if st.button("Start optimization process for ward"):
    st.session_state.start = True

if st.session_state.start:
    st.subheader("Choose target month")
    selected_month = st.selectbox("Select month", get_upcoming_months())

    st.subheader("Task Parameters")
    user_tasks = {}
    for i in range(1, 6):
        st.markdown(f"**Task {i}: {tasks[i]['name']}**")
        minutes = st.number_input(
            f"Minutes for Task {i}",
            value=float(tasks[i]['default_minutes']),
            min_value=0.0,
            key=f"min_{i}"
        )
        scales_with_area = st.radio(
            f"Scales with area?",
            options=["Yes", "No"],
            index=0 if tasks[i]["scales_with_area"] else 1,
            key=f"scale_{i}"
        )
        user_tasks[i] = {
            "minutes": minutes,
            "scales_with_area": scales_with_area == "Yes"
        }

    if st.button("Submit and calculate Ci"):
        # Save user input to log
        log_data = {
            "timestamp": datetime.datetime.now(),
            "month": selected_month,
            "ward_name": selected_ward_name,
            "ward_code": selected_ward_code
        }
        for i in user_tasks:
            log_data[f"task_{i}_minutes"] = user_tasks[i]["minutes"]
            log_data[f"task_{i}_scales_with_area"] = user_tasks[i]["scales_with_area"]

        log_df = pd.DataFrame([log_data])
        log_file = "user_input_log.csv"
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

        # Calculate Ci
        ward_df["Ci"] = ward_df.apply(lambda row: calculate_Ci(row, user_tasks), axis=1)

        # Clear previous inputs
        st.session_state.start = False

        # Display results page
        st.header(f"Results for {selected_ward_name} ({selected_ward_code})")
        st.dataframe(ward_df[["LSOA name", "ward name", "LSOA Area Size (HA)", "Ci"]])

        # Action buttons
        if st.button(f"✅ Optimize allocation of police officers for {selected_ward_name} with ward code {selected_ward_code}"):
            st.success("Optimization process initiated.")
            # Implement actual optimization logic here

        if st.button("❌ Cancel and restart process"):
            st.session_state.start = False
            st.session_state.clear()  # Clear session state
