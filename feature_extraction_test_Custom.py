import torch
from PIL import Image
from custom_clip_vision_encoder import CustomCLIPVisionEncoder  # ✅ 직접 만든 Vision Tower 임포트

# ✅ 1️⃣ CustomCLIPVisionEncoder 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_tower = CustomCLIPVisionEncoder(model_name="openai/clip-vit-large-patch14-336", 
                                       select_layer=-2, select_feature="patch").to(device)

# ✅ 2️⃣ Image Processor 가져오기
image_processor = vision_tower.image_processor  # ✅ CLIP Image Processor 가져오기

# ✅ 3️⃣ `expand2square()` 함수 추가 (load_images()와 동일한 방식)
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

# ✅ 4️⃣ `extract_image_features()` - `mm_projector` 적용 없이 저장
def extract_image_features(image_paths):
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # ✅ 1️⃣ 이미지 로드

        # ✅ 2️⃣ 정확한 `expand2square()` 적용 (load_images() 방식과 동일)
        background_color = [int(c * 255) for c in image_processor.image_mean]  
        image = expand2square(image, tuple(background_color))

        # ✅ 3️⃣ 정확한 `image_processor.preprocess()` 사용 (load_images()와 동일)
        inputs = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].squeeze(0).to(device)

        image_tensors.append(inputs)

    image_tensors = torch.stack(image_tensors, dim=0)  # ✅ 4️⃣ (M, 3, 336, 336) 형태로 변환

    # ✅ 5️⃣ CustomCLIPVisionEncoder 실행 (CLIP Vision Model Feature 추출)
    with torch.no_grad():
        image_features = vision_tower(image_tensors)  # ✅ `mm_projector` 적용 없이 저장

    return image_features  # ✅ `1024` 차원 Feature 저장

# ✅ 5️⃣ 저장할 데이터
image_paths = ["images/0.png", "images/1.png"]
import pdb;pdb.set_trace()
image_features = extract_image_features(image_paths).cpu()
torch.save(image_features, "new_image_features.pt")  # ✅ 1024차원 Feature 저장

print("✅ Feature 저장 완료! 저장 크기 최적화됨 🚀")
