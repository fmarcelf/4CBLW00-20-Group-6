import streamlit as st

st.title("What is Ci?")

st.markdown("""
### Understanding Ci (Police Time Allocation Index)

**Ci** represents the **number of police officers needed** to complete all assigned tasks within a specific Local Statistical Output Area (LSOA) during a fixed 2-hour timeslot.

### How Ci works

- Each LSOA has a fixed set of tasks with assigned minutes.
- The total minutes of all tasks represent the total police time needed to cover the area.
- Since shifts are 2 hours (120 minutes), dividing total minutes by 120 gives Ci — the number of officers required.
""")

st.markdown("**Example:**")
st.latex(r"Ci = \frac{\text{Total task time (minutes)}}{120} = \frac{240}{120} = 2")
st.markdown("Meaning 2 officers are needed in that 2-hour slot to complete all tasks. We always round up to the nearest integer number.")

st.markdown("""
### Why assigning more than Ci is not good

- Assigning **more officers than Ci** means some officers will have no tasks to do during the timeslot.
- This leads to **wasted resources and inefficiency** in police deployment.
- Our optimizer **ensures that no more than Ci officers are assigned** to any LSOA in a timeslot, avoiding idle manpower.

### What happens if fewer than Ci officers are assigned?

- Assigning fewer officers than Ci means not all tasks can be done in the timeslot.
- This reflects **partial coverage of the risk** associated with that LSOA.
- The optimizer accounts for this by incorporating the **reduced risk coverage into the objective function**, balancing resources across areas.

### How to assign minutes to tasks?

On the Optimization page, you specify minutes for each task that represent the maximum total time all police officers assigned to that LSOA during a 2-hour shift should collectively spend on that task. Default values are provided, which you can adjust as needed to better fit your operational needs.

---

Use the **Optimization** page to assign officers efficiently, respecting Ci to balance full task completion and resource use.
""")

