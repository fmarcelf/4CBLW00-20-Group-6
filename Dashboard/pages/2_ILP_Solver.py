import streamlit as st

st.title("ILP Solver")

st.markdown("""
### Objective
The goal of the ILP (Integer Linear Programming) solver is to assign a sufficient number of police officers to each **LSOA** (Lower Super Output Area) during specific time intervals to **maximize coverage of risky areas**.

The optimization is based on:
- A **standardized risk factor** per LSOA per time slot.
- The number of officers needed to cover an LSOA, which depends on its size and task demands (Ci).
- Optimization is per ward. 
""")

st.markdown("---")

st.subheader("Objective Function")
st.markdown("We aim to maximize the total standardized risk covered across all areas and time slots:")
st.latex(r"""
\text{Maximize} \quad \sum_{i \in I} \sum_{t \in T} R_{i,t} \cdot \frac{X_{i,t}}{C_i}
""")
st.markdown("Where:")
st.latex(r"R_{i,t} = \text{Standardized risk factor for LSOA } i \text{ at time } t")
st.latex(r"C_i = \text{Number of officers needed to fully cover LSOA } i \text{ during a 2-hour shift }")
st.latex(r"X_{i,t} = \text{Number of officers assigned to LSOA } i \text{ at time } t")

st.markdown("---")

st.subheader("Variables that have to be optimized")
st.latex(r"X_{i,t} \in \mathbb{N}_0")

st.markdown("---")

st.subheader("Constraints")
st.markdown("The model is subject to the following constraints:")
st.latex(r"\sum_{i \in I} \sum_{t \in T} X_{i,t} \leq 90")
st.latex(r"X_{i,t} \leq C_i \quad \forall i, t")
st.latex(r"\sum_{i \in I} X_{i,t} \geq 5 \quad \forall t")
st.latex(r"\sum_{t \in T} X_{i,t} \geq \left\lceil \frac{1}{7} \cdot C_i \right\rceil \quad \forall i")
st.markdown("""
These constraints ensure that officer deployment is realistic and balanced. The total officer limit (≤ 90) reflects resource availability, while per-LSOA limits prevent overstaffing beyond what each area requires. Minimum daily and per-time-slot allocations guarantee basic presence across all LSOAs and time periods. In addition to the model's daily officer assignment limit of 90, the police department retains flexibility to manually allocate the remaining 10 officer assignments, reducing reliance on the model’s decisions.
""")
st.markdown("---")

st.subheader("Definition of Covering an LSOA")
st.markdown("To compute the number of officers needed to cover LSOA \\(i\\), we estimate the total time required for all tasks in a 2-hour shift:")

st.markdown("Tasks per LSOA include:")
st.markdown("""
1. **Drive around the LSOA** (5 mins per hectare)  
2. **Engage with residents** (60 mins)  
3. **Foot patrol in key zones** (1 min per hectare)  
4. **Visit victims or vulnerable homes** (60 mins)  
5. **Other tasks** (e.g., write reports, check CCTV) (60 mins)

The values in between brackets are default values - hence they can be changed. Also whether or not it scales with area can be changed.
Use the **What is Ci** page to learn more about this. 

""")

st.markdown("---")

st.subheader("Data Preparation")
st.markdown("""
Crime data was sourced from [data.police.uk](https://data.police.uk/data/), and preprocessed using PowerQuery to harmonize historic and current LSOA codes.
""")
