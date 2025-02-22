import torch
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model

# ✅ 1️⃣ 모델 로드 (이제 CLIP과 T5 Tokenizer를 로드하지 않음!)
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_model = CLIPT5Model(model_name='clip-flant5-xxl', device=device)

# ✅ 2️⃣ 저장된 Feature 로드
image_features = torch.load("image_features.pt").to(device)  # ✅ (M, 576, 1024)
input_ids = torch.load("input_ids.pt").to(device)  # ✅ (N, sequence_length)
labels = torch.load("labels.pt").to(device)  # ✅ (N, sequence_length)


# ✅ 4️⃣ OOM 방지를 위해 for문을 사용한 개별 실행
def compute_vqa_scores(raw_images=None, img_features=None, raw_texts=None, input_ids=None, labels=None):
    """ 네 가지 입력 방식에 대해 VQAScore를 개별적으로 계산하여 OOM 방지 """
    M = len(image_features.shape[0])
    N = len(input_ids.shape[0])
    scores = torch.zeros((M, N)).to(device)  # 빈 score 텐서 생성


    for j in range(N):
        # ✅ text_feat을 딕셔너리 형태로 구성
        txt_feat = None if input_ids is None else {
            "input_ids": input_ids[j].unsqueeze(0),  # (1, sequence_length)
            "labels": labels[j].unsqueeze(0)  # (1, sequence_length)
        }

        score = vqa_model.forward(
            image_features=image_features,
            text_features=txt_feat,
        )
        scores[:, j] = score.item()

    return scores


# ✅ (4) 이미지 feature + Tokenized 텍스트
score_4 = compute_vqa_scores(img_features=image_features, input_ids=input_ids, labels=labels)
print(f"✅ (4) Image Feature + Tokenized Text Score:\n{score_4}")
