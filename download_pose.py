import sys
# Mock mediapipe to prevent crashes (just in case)
from unittest.mock import MagicMock
sys.modules["mediapipe"] = MagicMock()
sys.modules["mediapipe.python.solutions"] = MagicMock()

from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np

print("⏳ Initializing OpenPose (This will trigger model downloads)...")

try:
    # Use the stable OpenPose detector from lllyasviel
    model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    
    print("⚡ Running dummy prediction to cache models...")
    # include_hand=True and include_face=True ensures we get the full skeleton
    model(dummy_img, include_hand=True, include_face=True)
    
    print("✅ SUCCESS! OpenPose models are downloaded and ready.")

except Exception as e:
    print(f"❌ Error: {e}")