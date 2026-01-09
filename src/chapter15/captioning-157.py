# ------------------------------------------------------------
# Image Captioning using Salesforce BLIP (Hugging Face)
# ------------------------------------------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# ------------------------------------------------------------
# 1. Load Pretrained Model + Processor
# ------------------------------------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ------------------------------------------------------------
# 2. Load Input Image
# ------------------------------------------------------------
image_path = "cat.png"
image = Image.open(image_path).convert("RGB")

# ------------------------------------------------------------
# 3. Generate Caption
# ------------------------------------------------------------
inputs = processor(image, return_tensors="pt")  # Preprocess
out = model.generate(**inputs, max_length=50)   # Generate caption
caption = processor.decode(out[0], skip_special_tokens=True)

# ------------------------------------------------------------
# 4. Display Result
# ------------------------------------------------------------
print("🖼️ Image Caption:")
print(f"👉 {caption}")
