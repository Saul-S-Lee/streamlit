
import numpy as np
import pandas as pd
import streamlit as st

# this is a hello world app to test out streamlit

st.markdown("# Hello!")

# generate and display data
st.markdown("Generate data and display the dataframe")

col_names = ["A", "B"]
seed_data = [(1, 10), (2, 20), (3, 30), (4, 40)]
if st.button("Shuffle Data"):
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(4, 2)),
        columns=col_names,
    )
else:
    df = pd.DataFrame(
        seed_data,
        columns=col_names,
    )

st.write(df)

st.line_chart(df)
