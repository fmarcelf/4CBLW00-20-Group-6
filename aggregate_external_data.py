import pandas as pd

# ==============================================
# STEP 1: LOAD BURGLARY DATA (LOCAL PATHS)
# ==============================================
burglaries = pd.read_csv('data_aggregated_covid.cs')


# ==============================================
# STEP 2: PREPARE ACCOMMODATION PROPORTIONS (ROBUST)
# ==============================================
# 2a. Load the two sheets (2011 and 2021) from "accommodation type.xlsx"
accom_2011 = pd.read_excel('accommodation type.xlsx', sheet_name="2011")
accom_2021 = pd.read_excel('accommodation type.xlsx', sheet_name="2021")

# 2b. Rename columns for consistency
col_map_accom = {
    "LSOA code": "LSOA code",
    "All households ": "All_households",       # Note the trailing space
    "Detached": "Detached",
    "Semi-detached": "Semi_detached",
    "Terraced": "Terraced",
    "Purpose built flat": "Purpose_built_flat",
    "Flat in a converted/ shared house (includes all households in shared dwellings)": "Flat_converted_shared",
    "Flat in a commercial building": "Flat_commercial_building",
    "Caravan / other mobile or temporary structure": "Caravan_other"
}
accom_2011 = accom_2011.rename(columns=col_map_accom)
accom_2021 = accom_2021.rename(columns=col_map_accom)

# 2c. Define the seven dwelling-type columns
dwelling_types = [
    "Detached",
    "Semi_detached",
    "Terraced",
    "Purpose_built_flat",
    "Flat_converted_shared",
    "Flat_commercial_building",
    "Caravan_other"
]

# 2d. Compute raw proportions for each dwelling type in 2011 and 2021
for dt in dwelling_types:
    accom_2011[f"raw_prop_{dt}_2011"] = accom_2011[dt] / accom_2011["All_households"]
    accom_2021[f"raw_prop_{dt}_2021"] = accom_2021[dt] / accom_2021["All_households"]

# 2e. Keep only LSOA code + raw_prop columns, rename to unified "raw_prop_<dt>"
prop_cols_2011 = ["LSOA code"] + [f"raw_prop_{dt}_2011" for dt in dwelling_types]
prop_cols_2021 = ["LSOA code"] + [f"raw_prop_{dt}_2021" for dt in dwelling_types]

accom_raw_2011 = accom_2011[prop_cols_2011].rename(
    columns={f"raw_prop_{dt}_2011": f"raw_prop_{dt}" for dt in dwelling_types}
)
accom_raw_2021 = accom_2021[prop_cols_2021].rename(
    columns={f"raw_prop_{dt}_2021": f"raw_prop_{dt}" for dt in dwelling_types}
)

# 2f. Merge raw accommodation proportions onto burglary by Year cutoff
df_accom_pre2022 = burglaries[burglaries["Year"] <= 2021].merge(
    accom_raw_2011, on="LSOA code", how="left"
)
df_accom_post2021 = burglaries[burglaries["Year"] > 2021].merge(
    accom_raw_2021, on="LSOA code", how="left"
)

merged_accom = pd.concat([df_accom_pre2022, df_accom_post2021], ignore_index=True)

# 2g. Adjust raw_prop_<dt> into final prop_<dt> (3 decimals, sum=1.0) for accommodation
def adjust_accom_props(row):
    raw_vals = [row.get(f"raw_prop_{dt}") for dt in dwelling_types]
    if any(pd.isna(v) for v in raw_vals):
        # If any raw is missing, set all final props to NaN
        return pd.Series({f"prop_{dt}": pd.NA for dt in dwelling_types})
    # Round each raw to 3 decimals
    rounded = [round(v, 3) for v in raw_vals]
    total_rounded = sum(rounded)
    diff = round(1.0 - total_rounded, 3)
    # Add diff to the category with the largest raw proportion
    max_idx = int(pd.Series(raw_vals).idxmax())
    rounded[max_idx] = round(rounded[max_idx] + diff, 3)
    return pd.Series({f"prop_{dwelling_types[i]}": rounded[i] for i in range(len(dwelling_types))})

accom_props_df = merged_accom.apply(adjust_accom_props, axis=1)
merged_accom = pd.concat([merged_accom, accom_props_df], axis=1)
# Drop raw_prop columns for accommodation
merged_accom = merged_accom.drop(columns=[f"raw_prop_{dt}" for dt in dwelling_types])


# ==============================================
# STEP 3: PREPARE HOURS‐WORKED PROPORTIONS (ROBUST)
# ==============================================
# 3a. Load the “hours worked” Excel sheets
hours_2011 = pd.read_excel('hours worked.xlsx', sheet_name="2011")
hours_2021 = pd.read_excel('hours worked.xlsx', sheet_name="2021")

# 3b. Rename hours worked columns for consistency
hours_2011 = hours_2011.rename(columns={
    "All usual residents aged 16-74 in employment": "Total_employed_2011",
    "15 hours or less": "hrs_15_or_less",
    "16 to 30 hours": "hrs_16_30",
    "31 to 48 hours": "hrs_31_48",
    "49 or more hours": "hrs_49_more"
})
hours_2021 = hours_2021.rename(columns={
    "All usual residents aged 16+  in employment": "Total_employed_2021",
    "15 hours or less": "hrs_15_or_less",
    "16 to 30 hours": "hrs_16_30",
    "31 to 48 hours": "hrs_31_48",
    "49 or more hours": "hrs_49_more"
})

# 3c. Define the four hours categories
hours_categories = ["hrs_15_or_less", "hrs_16_30", "hrs_31_48", "hrs_49_more"]

# 3d. Compute raw proportions for each category in 2011 and 2021
for cat in hours_categories:
    hours_2011[f"raw_prop_{cat}_2011"] = hours_2011[cat] / hours_2011["Total_employed_2011"]
    hours_2021[f"raw_prop_{cat}_2021"] = hours_2021[cat] / hours_2021["Total_employed_2021"]

# 3e. Keep LSOA code + raw_prop columns, rename to unified "raw_prop_<cat>"
prop_cols_h2011 = ["LSOA code"] + [f"raw_prop_{cat}_2011" for cat in hours_categories]
prop_cols_h2021 = ["LSOA code"] + [f"raw_prop_{cat}_2021" for cat in hours_categories]

hw_raw_2011 = hours_2011[prop_cols_h2011].rename(
    columns={f"raw_prop_{cat}_2011": f"raw_prop_{cat}" for cat in hours_categories}
)
hw_raw_2021 = hours_2021[prop_cols_h2021].rename(
    columns={f"raw_prop_{cat}_2021": f"raw_prop_{cat}" for cat in hours_categories}
)

# 3f. Merge raw hours props onto merged_accom by year cutoff
df_hw_pre2022 = merged_accom[merged_accom["Year"] <= 2021].merge(
    hw_raw_2011, on="LSOA code", how="left"
)
df_hw_post2021 = merged_accom[merged_accom["Year"] > 2021].merge(
    hw_raw_2021, on="LSOA code", how="left"
)

merged_full = pd.concat([df_hw_pre2022, df_hw_post2021], ignore_index=True)

# 3g. Adjust raw_prop_<cat> into final prop_<cat> (3 decimals, sum=1.0) for hours worked
def adjust_hour_props(row):
    raw_vals = [row.get(f"raw_prop_{cat}") for cat in hours_categories]
    if any(pd.isna(v) for v in raw_vals):
        # If any raw is missing, final props = NaN
        return pd.Series({f"prop_{cat}": pd.NA for cat in hours_categories})
    # Round each raw to 3 decimals
    rounded = [round(v, 3) for v in raw_vals]
    total_rounded = sum(rounded)
    diff = round(1.0 - total_rounded, 3)
    # Add diff to category with largest raw proportion
    max_idx = int(pd.Series(raw_vals).idxmax())
    rounded[max_idx] = round(rounded[max_idx] + diff, 3)
    return pd.Series({f"prop_{hours_categories[i]}": rounded[i] for i in range(len(hours_categories))})

hour_props_df = merged_full.apply(adjust_hour_props, axis=1)
merged_full = pd.concat([merged_full, hour_props_df], axis=1)
# Drop raw_prop hours columns
merged_full = merged_full.drop(columns=[f"raw_prop_{cat}" for cat in hours_categories])


# ==============================================
# STEP 4: SAVE THE FINAL CSV
# ==============================================
output_path = 'burglaries_with_accom_and_hours_props.csv'
merged_full.to_csv(output_path, index=False)

print(f"✔︎ Final CSV saved to: {output_path}")



