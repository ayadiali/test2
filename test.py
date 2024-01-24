import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Streamlit app
st.title("Simple Streamlit App")

# Plot the data
st.line_chart(np.column_stack((x, y)))

# Display some text
st.write("This is a basic Streamlit app demonstrating a simple plot.")
