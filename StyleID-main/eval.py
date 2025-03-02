import torch
import lpips
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from transformers import CLIPImageProcessor, CLIPModel

# 加载图像文件
image1 = Image.open("tests/test/6/samples/0.jpg")
image2 = Image.open("output/batch_0/0_stylized_6_translated.jpg")
image2 = image1

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net="alex")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 计算LPIPS
image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
lpips_distance = lpips_model(image1_tensor, image2_tensor)
print("LPIPS distance:", lpips_distance.item())

# 计算CLIP score
image_a = preprocess(image1, return_tensors="pt")["pixel_values"]
image_b = preprocess(image2, return_tensors="pt")["pixel_values"]
embedding_a = clip_model.get_image_features(image_a)
embedding_b = clip_model.get_image_features(image_b)
clip_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
print("CLIP score:", clip_score.item())

# 计算SSIM
image1_gray = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
image2_gray = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
ssim = compare_ssim(image1_gray, image2_gray)
print("SSIM: ", ssim)