"""Live demo for the face verification model, deployed on Streamlit
Community Cloud.

Reuses ml-service/src/inference.py directly (same detection,
preprocessing, and prediction code the FastAPI service and its tests
use) rather than duplicating logic -- there's exactly one inference
implementation in this repo, this app just gives it a UI.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-service", "src"))

import numpy as np
import streamlit as st

from inference import NoFaceDetectedError, detect_and_crop_face, load_artifact, preprocess_face

st.set_page_config(page_title="Face Verification Demo", page_icon="🔍")

st.title("Face Verification Demo")
st.markdown(
    "PCA (Eigenfaces) + SVM classifier trained to distinguish "
    "**Arnold Schwarzenegger** from everyone else in the LFW dataset. "
    "Upload a photo containing a face to try it. "
    "[Source code](https://github.com/ShaheenaFalak123/facial-recognition-access-system)"
)


@st.cache_resource
def get_model():
    return load_artifact()


model, image_shape, threshold = get_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    import cv2

    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(uploaded, caption="Uploaded image", width=300)

    try:
        face_gray = detect_and_crop_face(image_bgr)
    except NoFaceDetectedError:
        st.error("No face detected in this image. Try a clearer, front-facing photo.")
    else:
        st.image(face_gray, caption="Detected face (as fed to the model)", width=150)

        X = preprocess_face(face_gray, image_shape)
        confidence = float(model.predict_proba(X)[0, 1])
        is_match = confidence >= threshold

        if is_match:
            st.success(f"Match: Arnold Schwarzenegger (confidence {confidence:.3f}, threshold {threshold:.3f})")
        else:
            st.info(f"No match (confidence {confidence:.3f}, threshold {threshold:.3f})")

        with st.expander("Why a tuned threshold instead of 0.5?"):
            st.markdown(
                "Schwarzenegger images are ~1.3% of the training data. "
                "The classifier's default 0.5 cutoff predicts 'not a match' "
                "for everything under that imbalance. The threshold above was "
                "instead chosen by maximizing F1 on out-of-fold cross-validated "
                "predictions -- see `ml-service/src/train.py` for the full "
                "methodology, including class-weighting, minority-class "
                "augmentation, and leakage-safe grouped cross-validation."
            )
