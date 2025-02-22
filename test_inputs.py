import torch
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model

# ✅ 1️⃣ 모델 로드 (이제 CLIP과 T5 Tokenizer를 로드하지 않음!)
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_model = CLIPT5Model(model_name='clip-flant5-xxl', device=device)

# ✅ 2️⃣ 저장된 Feature 로드
image_features = torch.load("image_features.pt").to(device)  # ✅ (M, 576, 1024)
input_ids = torch.load("input_ids.pt").to(device)  # ✅ (N, sequence_length)
labels = torch.load("labels.pt").to(device)  # ✅ (N, sequence_length)

# # ✅ 3️⃣ 원본 데이터 정의 (raw 이미지 경로 & 텍스트)
# image_paths = ["images/0.png", "images/1.png"]  # 2개의 테스트 이미지
# texts = ["someone talks on the phone angrily while another person sits happily",
#          "someone talks on the phone happily while another person sits angrily"]  # 2개의 테스트 텍스트

# ✅ 4️⃣ OOM 방지를 위해 for문을 사용한 개별 실행
def compute_vqa_scores(raw_images=None, img_features=None, raw_texts=None, input_ids=None, labels=None):
    """ 네 가지 입력 방식에 대해 VQAScore를 개별적으로 계산하여 OOM 방지 """
    # M = len(image_paths)
    # N = len(texts)
    M = len(image_features.shape[0])
    N = len(input_ids.shape[0])
    scores = torch.zeros((M, N)).to(device)  # 빈 score 텐서 생성

    for i in range(M):
        for j in range(N):
            img_feat = None if img_features is None else img_features[i].unsqueeze(0)  # (1, 576, 1024)
            
            # ✅ text_feat을 딕셔너리 형태로 구성
            txt_feat = None if input_ids is None else {
                "input_ids": input_ids[j].unsqueeze(0),  # (1, sequence_length)
                "labels": labels[j].unsqueeze(0)  # (1, sequence_length)
            }

            score = vqa_model.forward(
                # images=[raw_images[i]] if raw_images is not None else None,
                image_features=img_feat,
                # texts=[raw_texts[j]] if raw_texts is not None else None,
                text_features=txt_feat,
            )
            scores[i, j] = score.item()

    return scores

# ✅ (1) Raw 이미지 + Raw 텍스트
# score_1 = compute_vqa_scores(raw_images=image_paths, raw_texts=texts)
# print(f"✅ (1) Raw Image + Raw Text Score:\n{score_1}")

# # ✅ (2) 이미지 feature + Raw 텍스트
# score_2 = compute_vqa_scores(img_features=image_features, raw_texts=texts)
# print(f"✅ (2) Image Feature + Raw Text Score:\n{score_2}")

# # ✅ (3) Raw 이미지 + Tokenized 텍스트
# score_3 = compute_vqa_scores(raw_images=image_paths, input_ids=input_ids, labels=labels)
# print(f"✅ (3) Raw Image + Tokenized Text Score:\n{score_3}")

# ✅ (4) 이미지 feature + Tokenized 텍스트
score_4 = compute_vqa_scores(img_features=image_features, input_ids=input_ids, labels=labels)
print(f"✅ (4) Image Feature + Tokenized Text Score:\n{score_4}")

# ✅ 5️⃣ 모든 점수 비교
# print("\n🔎 비교 결과:")
# print(f"Raw Image + Raw Text == Image Feature + Raw Text? {'✅' if torch.allclose(score_1, score_2) else '❌'}")
# print(f"Raw Image + Raw Text == Raw Image + Tokenized Text? {'✅' if torch.allclose(score_1, score_3) else '❌'}")
# print(f"Raw Image + Raw Text == Image Feature + Tokenized Text? {'✅' if torch.allclose(score_1, score_4) else '❌'}")

