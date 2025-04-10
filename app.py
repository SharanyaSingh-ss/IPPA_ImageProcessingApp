import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
from filters import apply_grayscale, apply_gaussian_blur, apply_sepia, apply_sharpening, apply_cartoon_effect

# Set up Streamlit page
st.set_page_config(layout="wide")
st.title("Image Processing App")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image efficiently
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for correct color display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize large images for better performance
    max_size = 800  # Max display width
    h, w, _ = image.shape
    if w > max_size:
        scale_factor = max_size / w
        image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

    # Display original image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Processing image..."):
        # Sidebar: Image Transformations
        st.sidebar.header("Transformations")

        # Rotation
        rotation_angle = st.sidebar.selectbox("Rotate Image", [0, 90, 180, 270])
        if rotation_angle:
            image = cv2.rotate(image, {
                90: cv2.ROTATE_90_CLOCKWISE, 
                180: cv2.ROTATE_180, 
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }.get(rotation_angle, 0))

        # Flipping
        flip_type = st.sidebar.selectbox("Flip Image", ["None", "Horizontal", "Vertical", "Both"])
        if flip_type == "Horizontal":
            image = cv2.flip(image, 1)
        elif flip_type == "Vertical":
            image = cv2.flip(image, 0)
        elif flip_type == "Both":
            image = cv2.flip(image, -1)

        # Resizing
        st.sidebar.subheader("Resize Image")
        new_width = st.sidebar.slider("Width", 50, w, w)
        new_height = st.sidebar.slider("Height", 50, h, h)
        image = cv2.resize(image, (new_width, new_height))

        # Image Enhancement
        st.sidebar.header("Image Enhancement")

        # Brightness & Contrast
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0)
        enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        # Noise Removal
        noise_reduction = st.sidebar.selectbox("Noise Removal", ["None", "Gaussian", "Median"])
        if noise_reduction == "Gaussian":
            enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        elif noise_reduction == "Median":
            enhanced_image = cv2.medianBlur(enhanced_image, 5)

        # Filter selection
        filter_options = {
            "Grayscale": apply_grayscale,
            "Gaussian Blur": apply_gaussian_blur,
            "Sepia": apply_sepia,
            "Sharpening": apply_sharpening,
            "Cartoon Effect": apply_cartoon_effect
        }

        selected_filters = st.multiselect("Select Filters", list(filter_options.keys()))

        # Apply filters in sequence
        processed_image = enhanced_image.copy()
        for selected_filter in selected_filters:
            processed_image = filter_options[selected_filter](processed_image)

    # Display processed image
    with col2:
        st.image(processed_image, caption="Processed Image", use_container_width=True)

    # Convert OpenCV image (NumPy array) to PIL Image
    processed_pil = Image.fromarray(processed_image)

    # Save the PIL image as a byte stream
    image_bytes = io.BytesIO()
    processed_pil.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Streamlit download button
    st.download_button(
        label="Download Processed Image",
        data=image_bytes,
        file_name="processed_image.png",
        mime="image/png"
    )