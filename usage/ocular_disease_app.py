import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Streamlit Page Configuration
st.set_page_config(
    page_title="Ocular Disease Recognition",
    page_icon="👁️",
    layout="wide"
)

st.title("Ocular Disease Recognition with Grad-CAM")
st.markdown(
    """
    Upload a **retinal fundus image** to estimate ocular disease probabilities
    and visualize **Grad-CAM explanations** for model predictions.

    ⚠️ *For educational and research purposes only. Not for clinical use.*
    """
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease Labels
label_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

disease_map = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Pathological Myopia",
    "O": "Other Abnormalities"
}

# Load Model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(label_cols)
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "efficientnet_b0"
        / "efficientnet_b0_odir.pth"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.to(device)
    model.eval()
    return model

model = load_model()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        outputs[:, target_class].backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam[0].detach().cpu().numpy()

# Target layer for EfficientNet-B0
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# File Upload
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

# Inference + Grad-CAM UI
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Multi-label prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0]

    st.subheader("Disease Probability Estimates")

    for i, label in enumerate(label_cols):
        st.progress(float(probs[i]))
        st.write(f"**{disease_map[label]}**: {probs[i]:.2%}")

    # Grad-CAM Visualization
    st.subheader("Grad-CAM Visual Explanations")

    threshold = 0.5
    predicted_indices = (probs >= threshold).nonzero(as_tuple=True)[0]

    if len(predicted_indices) == 0:
        st.info("No disease detected above the confidence threshold.")
    else:
        for idx in predicted_indices:
            disease_code = label_cols[idx]
            disease_name = disease_map[disease_code]
            confidence = probs[idx].item()

            st.markdown(f"### {disease_name} ({confidence:.2%})")

            cam = gradcam.generate(input_tensor, idx)
            cam_resized = cv2.resize(cam, image.size)

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_resized),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(original_np, caption="Original", use_column_width=True)

            with col2:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

            with col3:
                st.image(
                    overlay,
                    caption=f"Overlay ({disease_name})",
                    use_column_width=True
                )

    st.markdown("---")
    st.caption(
        "Grad-CAM highlights retinal regions that most influenced each disease prediction. "
        "This tool is intended for research and educational demonstrations only."
    )

# Footer
st.markdown(
    """
    <hr>
    <center>
    Deep Learning–based Ocular Disease Recognition<br>
    EfficientNet-B0 + Grad-CAM
    </center>
    """,
    unsafe_allow_html=True
)
