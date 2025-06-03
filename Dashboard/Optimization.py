import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime
import calendar
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, PULP_CBC_CMD
import textwrap

# Load burglary data
df = pd.read_csv("Burglary Data - One row per LSOA CSV.csv")

tasks = {
    1: {"name": "Drive around the assigned LSOA", "default_minutes": 5, "scales_with_area": True},
    2: {"name": "Engage with residents", "default_minutes": 60, "scales_with_area": False},
    3: {"name": "Foot patrol", "default_minutes": 1, "scales_with_area": True},
    4: {"name": "Visits to burglary victims", "default_minutes": 60, "scales_with_area": False},
    5: {"name": "Other (e.g., CCTV, notes)", "default_minutes": 60, "scales_with_area": False}
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

st.title("👮 Police Allocation Optimizer")

df.columns = df.columns.str.strip()

st.subheader("Choose a ward to optimize")

if "ward name" in df.columns and "ward code" in df.columns:
    df["ward_display"] = df["ward name"] + " (" + df["ward code"] + ")"
    ward_display_list = df[["ward_display", "ward name", "ward code"]].drop_duplicates()
    ward_display_list = ward_display_list.dropna(subset=["ward_display"])  # remove NaNs
    ward_display_list["ward_display"] = ward_display_list["ward_display"].astype(str)
    unique_wards = ward_display_list["ward_display"].dropna().unique()
    selected_display = st.selectbox("Select a ward", ["-"] + sorted(unique_wards))

    if selected_display != "-":
        selected_row = ward_display_list[ward_display_list["ward_display"] == selected_display].iloc[0]
        selected_ward_name = selected_row["ward name"]
        selected_ward_code = selected_row["ward code"]
        ward_df = df[df["ward code"] == selected_ward_code].copy()
    else:
        selected_ward_name = selected_ward_code = None
        ward_df = pd.DataFrame()
else:
    st.error("Columns 'ward name' and/or 'ward code' not found.")
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
        st.markdown(f"*Task {i}: {tasks[i]['name']}*")
        minutes = st.number_input(f"Minutes for Task {i}", value=float(tasks[i]['default_minutes']), min_value=0.0, key=f"min_{i}")
        scales_with_area = st.radio(f"Scales with area/per hectare?", ["Yes", "No"], index=0 if tasks[i]["scales_with_area"] else 1, key=f"scale_{i}")
        user_tasks[i] = {"minutes": minutes, "scales_with_area": scales_with_area == "Yes"}

    if st.button(f"✅ Optimize allocation for {selected_ward_name} ({selected_ward_code})"):
        with st.spinner("Solving ILP and generating report..."):
            import time
            time.sleep(3)

            ward_df["Ci"] = ward_df.apply(lambda row: calculate_Ci(row, user_tasks), axis=1)
            lsoa_ids = ward_df["LSOA code"].tolist()
            ci_values = dict(zip(ward_df["LSOA code"], ward_df["Ci"]))
            lsoa_names = dict(zip(ward_df["LSOA code"], ward_df["LSOA name"]))

            shifts = list(range(8))
            shift_labels = ["06–08", "08–10", "10–12", "12–14", "14–16", "16–18", "18–20", "20–22"]
            shift_weights = [0.8, 0.9, 0.95, 1.0, 1.1, 1.15, 1.2, 1.3]
            risk_factor = {(i, t): 1.25 * shift_weights[t] for i in lsoa_ids for t in shifts}

            prob = LpProblem("Officer_Allocation", LpMaximize)
            X = {(i, t): LpVariable(f"x_{i}_{t}", lowBound=0, cat=LpInteger) for i in lsoa_ids for t in shifts}
            prob += lpSum(risk_factor[i, t] * X[i, t] / ci_values[i] for i in lsoa_ids for t in shifts)
            prob += lpSum(X[i, t] for i in lsoa_ids for t in shifts) <= 90
            for i in lsoa_ids:
                for t in shifts:
                    prob += X[i, t] <= ci_values[i]
            for t in shifts:
                prob += lpSum(X[i, t] for i in lsoa_ids) >= 5
            for i in lsoa_ids:
                prob += lpSum(X[i, t] for t in shifts) >= math.ceil(ci_values[i] / 7)

            prob.solve(PULP_CBC_CMD(msg=False))

            allocation_result = {
                i: [int(X[i, t].varValue) if X[i, t].varValue else 0 for t in shifts]
                for i in lsoa_ids
            }

            now = datetime.datetime.now()
            selected_month_index = list(calendar.month_name).index(selected_month)
            display_year = now.year + 1 if now.month > selected_month_index else now.year
            month_year_label = f"{selected_month} {display_year}"
            pdf_filename = f"{selected_ward_name.replace(' ', '_')}_{selected_ward_code}_{selected_month}_{display_year}.pdf"

            with PdfPages(pdf_filename) as pdf:
                # Page 1
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis('off')
                ax.text(0.5, 0.65, f"Officer Allocation for {selected_ward_name} ({selected_ward_code})",
                        fontsize=20, ha='center', va='center', weight='bold')
                ax.text(0.5, 0.55, month_year_label, fontsize=18, ha='center', va='center')
                pdf.savefig(fig)
                plt.close(fig)

                # Page 2
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis('off')
                ax.set_title(f"LSOAs Overview for {selected_ward_name}", fontsize=18, loc='center', pad=20)
                table_data = ward_df[["LSOA name", "ward name", "LSOA Area Size (HA)", "Ci"]].head(20)
                col_labels = ["LSOA Name", "Ward Name", "Area Size (HA)", "Officer Allocation (Ci)"]
                table = ax.table(cellText=table_data.values, colLabels=col_labels,
                                 cellLoc='center', loc='upper left', bbox=[0.05, 0.4, 0.9, 0.5])
                for key, cell in table.get_celld().items():
                    if key[0] == 0:
                        cell.set_text_props(weight='bold', fontsize=10)
                    else:
                        cell.set_fontsize(10)
                pdf.savefig(fig)
                plt.close(fig)

                import textwrap

                # Page 3 - Shift allocation chessboard table
                fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 Landscape
                ax.axis('off')
                ax.set_title(f"Officer Allocation for {selected_ward_name} per shift", fontsize=18, pad=20,
                             loc='center')


                # Wrap LSOA names to fit in table rows
                def wrap_text(text, width=25):
                    return '\n'.join(textwrap.wrap(text, width))


                row_labels = [wrap_text(f"{lsoa_names[i]}\n({i})", width=30) for i in lsoa_ids]

                table_data = []
                for i in lsoa_ids:
                    row = []
                    for t in shifts:
                        val = allocation_result[i][t]
                        row.append(f"{val}" if val > 0 else "")
                    table_data.append(row)

                # Create the table with adjusted bounding box
                table = ax.table(
                    cellText=table_data,
                    rowLabels=row_labels,
                    colLabels=shift_labels,
                    cellLoc='center',
                    rowLoc='center',
                    loc='upper center',
                    bbox=[0.19, 0.20, 0.9, 0.75]  # [left, bottom, width, height]
                )

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.0, 1.2)

                # Bold header row
                for key, cell in table.get_celld().items():
                    if key[0] == 0 or key[1] == -1:
                        cell.set_text_props(weight='bold')

                pdf.savefig(fig)
                plt.close(fig)

                # Page 4 - Interpretation
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis('off')
                ax.set_title("What to do with these results", fontsize=18, pad=20)

                text = (
                    "The presented allocation results specify the recommended number of officers "
                    "to be assigned to each LSOA within the selected ward across the defined shifts. "
                    "These allocations are derived from historical crime data and are intended to guide "
                    "strategic deployment of police resources.\n\n"

                    "It is important to note that the total number of police officers assigned to all LSOAs "
                    "within a ward on any given day must not exceed 90. This constraint ensures that "
                    "the model's recommendations remain within realistic staffing limits.\n\n"

                    "An additional reserve of 10 officers per ward is maintained outside the model's allocation. "
                    "These officers can be flexibly deployed by local police stations to address unforeseen incidents, "
                    "emerging crime patterns, or situational demands that may not be captured in the model’s static input. "
                    "This operational flexibility ensures that police stations retain the ability to respond dynamically, "
                    "without being entirely dependent on model outputs."
                )

                # Center horizontally by using x=0.5 and setting ha='center'
                ax.text(
                    0.5, 0.9, text,
                    va='top', ha='center',
                    wrap=True, fontsize=12,
                    transform=ax.transAxes
                )

                pdf.savefig(fig)
                plt.close(fig)

        st.success("✅ PDF report generated successfully!")
        st.download_button("📄 Download PDF Report", data=open(pdf_filename, "rb").read(),
                           file_name=pdf_filename, mime="application/pdf")

        if st.button("❌ Cancel and restart"):
            st.session_state.start = False
            st.session_state.clear()
