import pickle
import cv2
import streamlit as st
from PIL import Image
import imagehash
import os

from B1_detect_person import detect_person
from B2_detect_keypoints import detect_keypoints
from B3_classify_poses import classify_poses

# Set Streamlit page title and icon
st.set_page_config(
    page_title="Gesture Classifier",
    page_icon=":camera:",
)

st.title("Gesture Classifier 🚀")

st.write("Instructions:")
st.write("1. Upload an image or select a predefined image:")
st.write("2. Click 'Save Image' to add the image in the pipeline")
st.write("3. Click 'Classify Poses' Button to perform the classification")
st.write("4. Click 'Display Images' to display the extracted images with their classified gestures")

# Define a function to calculate the perceptual hash of an image
def calculate_hash(image):
    pil_image = Image.fromarray(image)
    return str(imagehash.average_hash(pil_image))

# List of predefined test images in the "test_imgs" folder
test_images = ['img_1.jpg', 'img_2.jpg', 'img_3.jpg', 'img_4.jpg', 'img_5.jpg', 'img_6.jpg']

# Upload an image or select from predefined test images
uploaded_image = st.file_uploader("Upload an image or select a predefined image:", type=["jpg", "jpeg", "png"])

if not uploaded_image:
    # Allow the user to select a predefined test image
    selected_image = st.selectbox("Select a predefined image", test_images)

    if selected_image:
        uploaded_image = os.path.join("test_imgs", selected_image)

if uploaded_image is not None:
    # Display the uploaded image
    if isinstance(uploaded_image, str):
        st.image(uploaded_image, caption="Selected Image", use_column_width=True)
    else:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Save Image"):
        # If the selected image is predefined, copy it to the working directory
        if isinstance(uploaded_image, str):
            with open(os.path.join("./", 'img.jpg'), "wb") as f:
                with open(uploaded_image, 'rb') as source_image:
                    f.write(source_image.read())
        # For uploaded images, save them directly
        else:
            with open(os.path.join("./", 'img.jpg'), "wb") as f:
                f.write(uploaded_image.read())
        st.success("Processed image saved as 'img.jpg'")

# Add a button to run the script
if st.button("Classify gestures"):
    
    st.text("Detecting Persons...")
    detect_person()
    st.text("Detecting Keypoints for pose...")
    detect_keypoints()
    st.text("Classifying the gestures...")
    classify_poses()
    st.text("All functions executed successfully. Click on Display Images to view the results")

if st.button("Display Images"):
    # Load saved pickle file
    with open('data/image_data.pickle', 'rb') as f:
        image_dicts = pickle.load(f)

    # Set a threshold for hash similarity (lower values make it stricter)
    # Adjust this value as needed
    hash_similarity_threshold = 11
    flag = True

    # Create a set to store the hashes of displayed images
    displayed_hashes = set()

    for d in image_dicts:
        if d['pred'] is not None:
            current_image = d['image']
            current_hash = calculate_hash(current_image)

            # Check if the hash is not in the set of displayed hashes
            is_similar = False
            for displayed_hash in displayed_hashes:
                hamming_distance = imagehash.hex_to_hash(current_hash) - imagehash.hex_to_hash(displayed_hash)
                if hamming_distance <= hash_similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                flag = False
                displayed_hashes.add(current_hash)

                # Create a layout with two columns: one for the image and one for the text
                col1, col2 = st.columns(2)
                
                # Display the image in the first column
                col1.image(current_image, use_column_width=True)

                # Display the text (classification) in the second column
                col2.markdown(f"<h3>{d['pred']}</h3>", unsafe_allow_html=True)  # Increase the text size


    
    if flag:
        st.write("No poses detected!")
