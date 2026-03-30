import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st



# page setup
st.set_page_config(page_title="TBAD Segmentation Dashboard", layout="wide")
st.title("TBAD CT Segmentation Dashboard")
st.caption("Per-slice comparison across datasets")



# paths
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "dashboard_data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

DATASET_ORDER = ["Dataset100", "Dataset101", "Dataset102", "Dataset103"]



# load manifest
@st.cache_data #tells streamlit to only run this function once and cache the result, don't read the file multiple times
def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


manifest = load_manifest()



# helpers
def normalise_to_uint8(arr):
    """
    Scales any float or int array to the 0-255 range that image viewers expect
    CT scans 
    """
    arr = arr.astype(np.float32)
    if arr.max() > arr.min():
        # min-max normalisation: stretches the value range to exactly 0.0–1.0
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        # flat array (all same value) avoid divide-by-zero, just return zeros
        arr = np.zeros_like(arr)
    return (arr * 255).astype(np.uint8)


def load_image(path):
    # convert("L") forces grayscale — CT slices are single-channel images
    return normalise_to_uint8(np.array(Image.open(path).convert("L")))


def load_mask(path):
    # Mask pixels hold label integers (0,1,2,3), not visual brightness values,
    # so we skip normalisation and just read the raw values as uint8
    return np.array(Image.open(path).convert("L")).astype(np.uint8)


def make_rgb(gray):
    """
    Converts a 2D grayscale array (H, W) to a 3D RGB array (H, W, 3) by
    duplicating the single channel three times. Required before we can add
    colour overlays, since colour images need R, G, and B channels.
    """
    return np.stack([gray, gray, gray], axis=-1)


def blend_overlay(ct, mask, alpha=0.45):
    #    Blends coloured label regions on top of the greyscale CT image, alpha controls how much the colour overlay is visible (0 = invisible, 1 = solid).
    
    base = make_rgb(ct).astype(np.float32)
    #start with a blank overlay
    overlay = np.zeros_like(base, dtype=np.float32)

    # Paint each label region its assigned colour
    overlay[mask == 1] = [255, 0, 0]   # red = TL
    overlay[mask == 2] = [0, 0, 255]   # blue = FL
    overlay[mask == 3] = [0, 255, 0]   # green = FLT

    mask_present = (mask > 0)[..., None]
    # Where a label exists: mix base CT and colour overlay using alpha blending
    # Where no label:keep the original CT pixel unchanged
    out = np.where(mask_present, (1 - alpha) * base + alpha * overlay, base)
    return out.astype(np.uint8)


def dice(gt, pred):
    """
    Computes the Dice score — the standard metric for segmentation overlap.
    Ranges from 0 (no overlap) to 1 (perfect overlap).
 
    Formula: (2 x intersection) / (total pixels in gt + total pixels in pred)
 
    Special case: if both gt and pred are completely empty on this slice,
    the model correctly predicted nothing — score is 1.0 (perfect).
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # Count pixels where both gt AND pred are True (the overlap)
    inter = np.logical_and(gt, pred).sum()
    g = gt.sum()    # total labelled pixels in ground truth
    p = pred.sum()  # total labelled pixels in prediction
 

    if g == 0 and p == 0:
        return 0.0

    return (2 * inter) / (g + p)


def compute_slice_metrics(gt, pred):
    """
    Runs Dice for each label class separately, plus an overall foreground score.
    Passing boolean comparisons (gt == 1, pred == 1) isolates one class at a time.
    """
    return {
        "foreground": dice(gt > 0, pred > 0), # any label vs any label
        "TL": dice(gt == 1, pred == 1), # True Lumen only
        "FL": dice(gt == 2, pred == 2), # False Lumen only
        "FLT": dice(gt == 3, pred == 3), # False Lumen Thrombus only
    }


def format_val(v):
    # Rounds to 4 decimal places for consistent display in the UI
    return f"{v:.4f}"


def get_slice_info(dataset, case, slice_id):
    # Looks up the file paths for a specific slice from the manifest dictionary
    return manifest[dataset][case]["slices"][slice_id]


def file_path(rel):
    # Converts a relative path from the manifest into an absolute path
    return APP_DIR / rel



# sidebar
st.sidebar.header("Controls")

show_prediction_overlay = st.sidebar.toggle("Prediction overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Overlay colours**")
st.sidebar.markdown("Red = True Lumen")
st.sidebar.markdown("Blue = False Lumen")
st.sidebar.markdown("Green = False Lumen Thrombus")


# layout
# Creates one column per dataset, displayed side by side for direct comparison
cols = st.columns(len(DATASET_ORDER))

for i, dataset in enumerate(DATASET_ORDER):
    with cols[i]:
        st.subheader(dataset)

        cases = list(manifest[dataset].keys())
         # key= makes each selectbox unique so Streamlit doesn't confuse them
        case = st.selectbox(f"Case ({dataset})", cases, key=f"case_{dataset}")

        slices = list(manifest[dataset][case]["slices"].keys())
        slice_id = st.selectbox(f"Slice ({dataset})", slices, key=f"slice_{dataset}")

        info = get_slice_info(dataset, case, slice_id)

        img_path = file_path(info["image"])
        gt_path = file_path(info["gt"])
        pred_path = file_path(info["pred"])

        img = load_image(img_path)
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        # show CT or CT + prediction overlay
        if show_prediction_overlay:
            display = blend_overlay(img, pred)
            st.caption(f"{dataset} | {case} | slice {slice_id} | prediction overlay")
        else:
            display = img
            st.caption(f"{dataset} | {case} | slice {slice_id} | CT")

        st.image(display, use_container_width=True)

        # per-slice metrics 
        st.markdown("**Slice Metrics**")
        m = compute_slice_metrics(gt, pred)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Foreground: {format_val(m['foreground'])}")
            st.write(f"TL: {format_val(m['TL'])}")
        with c2:
            st.write(f"FL: {format_val(m['FL'])}")
            st.write(f"FLT: {format_val(m['FLT'])}")

        if not np.any(gt == 3) and not np.any(pred == 3):
            st.caption("FLT = 0.0 because both GT and prediction are empty on this slice.")
