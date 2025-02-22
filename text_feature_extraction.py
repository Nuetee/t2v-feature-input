import torch
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model, format_question, format_answer
from t2v_metrics.models.vqascore_models.mm_utils import t5_tokenizer_image_token

# ✅ 1️⃣ CLIPT5Model 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_model = CLIPT5Model(model_name='clip-flant5-xxl', device=device)
tokenizer = vqa_model.tokenizer  # ✅ 모델에서 사용하는 T5Tokenizer 가져오기

# ✅ 2️⃣ 질문 & 답변 템플릿 가져오기
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

# ✅ 3️⃣ 저장할 원본 텍스트 데이터
texts = ["someone talks on the phone angrily while another person sits happily",
         "someone talks on the phone happily while another person sits angrily"] 

# ✅ 4️⃣ `question_template`, `answer_template` 적용 (CLIPT5의 `forward()`와 동일)
questions = [default_question_template.format(text) for text in texts]
answers = [default_answer_template.format(text) for text in texts]

# ✅ 5️⃣ `format_question()`, `format_answer()` 적용 (CLIPT5의 `forward()`와 동일)
questions = [format_question(q, conversation_style=vqa_model.conversational_style) for q in questions]
answers = [format_answer(a, conversation_style=vqa_model.conversational_style) for a in answers]

# ✅ 6️⃣ T5 토크나이징 수행 (CLIPT5의 `forward()`와 동일)
input_ids = [t5_tokenizer_image_token(qs, tokenizer, return_tensors="pt") for qs in questions]
labels = [t5_tokenizer_image_token(ans, tokenizer, return_tensors="pt") for ans in answers]

# ✅ 7️⃣ 패딩을 고려하여 시퀀스 길이를 맞춤 (CLIPT5의 `forward()`와 동일)
input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # `IGNORE_INDEX` 사용

# ✅ 8️⃣ `input_ids.pt`와 `labels.pt` 별도로 저장
torch.save(input_ids, "input_ids.pt")
torch.save(labels, "labels.pt")

print("✅ `input_ids.pt` & `labels.pt` 저장 완료! CLIPT5Model과 동일한 전처리 적용됨 🚀")
