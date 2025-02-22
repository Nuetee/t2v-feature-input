import torch
from PIL import Image
from transformers import CLIPProcessor, T5Tokenizer
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model

# ✅ 1️⃣ CLIPT5Model 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_model = CLIPT5Model(model_name='clip-flant5-xxl', device=device)

# ✅ 2️⃣ CLIP Processor 로드 (이미지 전처리용)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

# ✅ 3️⃣ `encode_images()`를 직접 호출하여 Feature 추출
def extract_image_features(image_paths):
    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)  # ✅ 이미지 텐서 변환
        image_tensors.append(inputs)

    # ✅ `encode_images()` 직접 호출
    image_tensors = torch.cat(image_tensors, dim=0)  # (M, 3, 336, 336)
    image_tensors = image_tensors.to(dtype=torch.bfloat16)  # ✅ BFloat16 변환 추가!
    image_features = vqa_model.model.encode_images(image_tensors)  # ✅ encode_images() 호출
    return image_features  # ✅ (M, 576, 4096)

# ✅ 3️⃣ 텍스트 토큰화 함수
def tokenize_texts(texts):
    tokenized_texts = [tokenizer(text, return_tensors="pt")["input_ids"] for text in texts]
    tokenized_texts = torch.nn.utils.rnn.pad_sequence(
        tokenized_texts, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return tokenized_texts  # (N, sequence_length)

# ✅ 4️⃣ 저장할 데이터
image_paths = ["images/0.png", "images/1.png"]  # 2개의 테스트 이미지
texts = ["someone talks on the phone angrily while another person sits happily",
         "someone talks on the phone happily while another person sits angrily"]  # 2개의 테스트 텍스트

# ✅ 5️⃣ feature 추출 및 저장
image_features = extract_image_features(image_paths).to("cpu")  # ✅ GPU 메모리 절약 위해 CPU로 저장
torch.save(image_features, "image_features.pt")  # ✅ shape: (M, 576, 4096)

text_tokens = tokenize_texts(texts).to("cpu")
torch.save(text_tokens, "text_tokens.pt")  # ✅ shape: (N, sequence_length)

print("✅ Feature 저장 완료! 이제 메모리를 절약하며 실행 가능 🚀")

