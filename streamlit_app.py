# Import the json module so we can read the model metadata file.
import json
# Import Path so we can build file paths reliably.
from pathlib import Path

# Import NumPy so we can turn uploaded images into arrays.
import numpy as np
# Import pandas so we can build a tiny table for the probability chart.
import pandas as pd
# Import Streamlit for the web application interface.
import streamlit as st
# Import PyTorch so we can load and run the scripted model.
import torch
# Import PIL so we can open uploaded image files.
from PIL import Image, UnidentifiedImageError

# Build the base directory path from the location of this Python file.
BASE_DIR = Path(__file__).resolve().parent
# Build the path to the models folder.
MODEL_DIR = BASE_DIR / "models"
# Build the path to the exported TorchScript model file.
SCRIPTED_MODEL_PATH = MODEL_DIR / "best_model_scripted.pt"
# Build the path to the metadata JSON file.
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Set the Streamlit page title, icon, and layout.
st.set_page_config(page_title="Chest X-Ray Pneumonia Classifier", page_icon="🫁", layout="centered")

# Define a cached function so the model loads only once per app session.
@st.cache_resource
# Start the model-loading function definition.
def load_model_assets():
    # Stop the app early if the scripted model file is missing.
    if not SCRIPTED_MODEL_PATH.exists():
        # Show a clear error message to the user.
        st.error("The file models/best_model_scripted.pt was not found.")
        # Stop Streamlit execution because the app cannot continue without the model.
        st.stop()

    # Stop the app early if the metadata file is missing.
    if not METADATA_PATH.exists():
        # Show a clear error message to the user.
        st.error("The file models/model_metadata.json was not found.")
        # Stop Streamlit execution because the app cannot continue without metadata.
        st.stop()

    # Open the metadata JSON file for reading.
    with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        # Load the metadata dictionary from JSON.
        metadata = json.load(metadata_file)

    # Load the TorchScript model on the CPU so the app runs on normal Streamlit hosting.
    model = torch.jit.load(str(SCRIPTED_MODEL_PATH), map_location="cpu")
    # Put the model into evaluation mode so dropout or batch norm stays stable.
    model.eval()
    # Return both the model and the metadata dictionary.
    return model, metadata

# Define a preprocessing function that matches the training pipeline.
def preprocess_image(image: Image.Image, image_size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    # Convert the uploaded image to RGB so the tensor has three channels.
    image = image.convert("RGB")
    # Resize the image to the same size used during model training.
    image = image.resize((image_size, image_size))
    # Convert the image into a NumPy array of type float32 and scale pixels to 0 to 1.
    image_array = np.array(image).astype(np.float32) / 255.0
    # Reorder the array from height-width-channel to channel-height-width.
    image_array = np.transpose(image_array, (2, 0, 1))
    # Turn the NumPy array into a PyTorch tensor and add a batch dimension.
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    # Build a tensor for the channel-wise mean values.
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    # Build a tensor for the channel-wise standard deviation values.
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    # Normalize the image tensor using the same values used during training.
    image_tensor = (image_tensor - mean_tensor) / std_tensor
    # Return the final input tensor ready for model inference.
    return image_tensor

# Define a helper function that runs inference on one uploaded image.
def predict_image(image: Image.Image, model, metadata: dict):
    # Read the image size from metadata so the app stays consistent with training.
    image_size = int(metadata["image_size"])
    # Read the normalization mean from metadata.
    mean = metadata["mean"]
    # Read the normalization standard deviation from metadata.
    std = metadata["std"]
    # Read the class names from metadata.
    class_names = metadata["class_names"]

    # Preprocess the uploaded image into a model-ready tensor.
    input_tensor = preprocess_image(image, image_size=image_size, mean=mean, std=std)

    # Turn off gradients because we are only doing inference.
    with torch.no_grad():
        # Run the model forward on the input tensor.
        outputs = model(input_tensor)
        # Convert raw model outputs into probabilities.
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        # Find the index of the class with the highest probability.
        predicted_index = int(torch.argmax(probabilities).item())
        # Read the class name for the winning index.
        predicted_label = class_names[predicted_index]
        # Read the confidence score for the winning class.
        confidence = float(probabilities[predicted_index].item())

    # Return the predicted label, confidence, and all probabilities.
    return predicted_label, confidence, probabilities.numpy().tolist()

# Write the main title at the top of the app.
st.title("Chest X-Ray Pneumonia Detection Demo")
# Add a short description below the title.
st.write(
    "Upload a chest X-ray image and the model will predict whether it looks more like NORMAL or PNEUMONIA."
)

# Add an expandable note so your audience sees the model limitation.
with st.expander("Important note"):
    # Explain that the tool is for educational demonstration only.
    st.write(
        "This app is an educational project demo. It is not a medical device and must not be used as a clinical diagnosis tool."
    )

# Load the model and metadata once near the top of the app.
model, metadata = load_model_assets()

# Ask the user to upload an image file.
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

# Run the prediction block only after a file has been uploaded.
if uploaded_file is not None:
    # Try to open the uploaded file as an image.
    try:
        # Read the uploaded file into a PIL image object.
        uploaded_image = Image.open(uploaded_file)
    # Catch errors when the uploaded file is not a valid image.
    except UnidentifiedImageError:
        # Show a helpful error message in the app.
        st.error("The uploaded file could not be read as an image. Please upload a JPG, JPEG, or PNG file.")
        # Stop further processing for this run.
        st.stop()

    # Show the uploaded image on the page so the user can confirm it.
    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    # Add a button so the user can control when inference runs.
    if st.button("Run prediction"):
        # Run the model on the uploaded image and collect the outputs.
        predicted_label, confidence, probabilities = predict_image(uploaded_image, model, metadata)

        # Write a section heading for the result.
        st.subheader("Prediction result")

        # Show a red box when the predicted class is pneumonia.
        if predicted_label == "PNEUMONIA":
            # Display the final predicted label in an error-style box.
            st.error(f"Predicted class: {predicted_label}")
        # Show a green box when the predicted class is normal.
        else:
            # Display the final predicted label in a success-style box.
            st.success(f"Predicted class: {predicted_label}")

        # Display the confidence score for the final predicted class.
        st.write(f"Confidence: {confidence:.4f}")

        # Build a small DataFrame to display class probabilities neatly.
        probability_frame = pd.DataFrame(
            {
                "Class": metadata["class_names"],
                "Probability": probabilities,
            }
        )

        # Add a heading for the probability chart.
        st.subheader("Class probabilities")
        # Show the probability table so the user can read exact values.
        st.dataframe(probability_frame, use_container_width=True)
        # Show a bar chart of the class probabilities.
        st.bar_chart(probability_frame.set_index("Class"))
# Show a helpful message before any file is uploaded.
else:
    # Prompt the user to upload an image to begin.
    st.info("Upload a chest X-ray image to start the demo.")
