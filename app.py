import streamlit as st

st.set_page_config(page_title="TBAD Segmentation Dashboard", layout="wide")

st.title("Thoracic Aortic Dissection Segmentation Dashboard")
st.write("nnU-Net 2D medical image segmentation demo")

st.sidebar.header("Controls")
case = st.sidebar.selectbox("Select Case", ["TBAD_101", "TBAD_102", "TBAD_103"])
model = st.sidebar.selectbox("Select Model", ["Baseline", "Improved", "Preprocessed"])
view = st.sidebar.selectbox("View Mode", ["CT Only", "Ground Truth", "Prediction", "Overlay"])

left, centre, right = st.columns([1, 2, 1])

with left:
    st.subheader("Info")
    st.write(f"Case: {case}")
    st.write(f"Model: {model}")
    st.write(f"View: {view}")

with centre:
    st.subheader("Image Viewer")
    st.info("Your CT image and mask display will go here.")

with right:
    st.subheader("Metrics")
    st.metric("Dice Score", "0.62")
    st.metric("Inference Time", "0.08s")
    st.write("Model: nnU-Net 2D")