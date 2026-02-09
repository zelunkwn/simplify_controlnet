from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import os

# --- CHANGE THESE TWO LINES ---
model_id = "LiheYoung/depth-anything-large-hf"  # Changed from 'small' to 'large'
save_directory = "./local_depth_model_large"    # New folder name

print(f"⏳ Downloading LARGE model to {save_directory}...")
print("⚠️ This is ~1.5 GB. It will take a while.")

try:
    processor = AutoImageProcessor.from_pretrained(model_id)
    processor.save_pretrained(save_directory)
    
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.save_pretrained(save_directory)
    
    print("✅ SUCCESS! Large model saved.")

except Exception as e:
    print(f"❌ Failed: {e}")