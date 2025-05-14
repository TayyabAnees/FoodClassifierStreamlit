import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import streamlit as st

# Directory to save uploaded files
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load model
@st.cache_resource
def load_model():
    model = models.googlenet(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("googlenet_food_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Prediction function
def predict_image(image):
    time.sleep(4)  # Simulate processing delay
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    class_names = ["Non-Food", "Food"]
    return class_names[pred_class.item()], confidence.item() * 100


# Streamlit UI
st.title("Food vs Non-Food Image Classifier")

upload_mode = st.radio("Choose Upload Mode:", ["Single Image", "Multiple Images (Simulated Folder Upload)"])

if upload_mode == "Single Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence = predict_image(image)
        st.success(f"Prediction: {label} ({confidence:.2f}%)")

elif upload_mode == "Multiple Images (Simulated Folder Upload)":
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            try:
                image = Image.open(file).convert("RGB")
                label, confidence = predict_image(image)
                results.append((file.name, label, confidence))
            except Exception as e:
                st.warning(f"Could not process {file.name}: {e}")

        st.subheader("Results:")
        for fname, label, conf in results:
            st.write(f"**{fname}** — {label} ({conf:.2f}%)")

