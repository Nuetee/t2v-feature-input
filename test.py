import t2v_metrics
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

### For a single (image, text) pair
image = "images/0.png" # an image path in string format
text = "someone talks on the phone angrily while another person sits happily"
score = clip_flant5_score(images=[image], texts=[text])
print(score)
### Alternatively, if you want to calculate the pairwise similarity scores 
### between M images and N texts, run the following to return a M x N score tensor.
# images = ["images/0.png", "images/1.png"]
# texts = ["someone talks on the phone angrily while another person sits happily",
#          "someone talks on the phone happily while another person sits angrily"]
image_features = torch.load("image_features.pt").to(device)  # ✅ (M, 576, 1024)
input_ids = torch.load("input_ids.pt").to(device)  # ✅ (N, sequence_length)
labels = torch.load("labels.pt").to(device)  # ✅ (N, sequence_length)
image_feature_list = []
input_id_list = []
label_list = []

M = image_features.shape[0]
N = input_ids.shape[0]

for j in range(M):
    image_feature = image_features[j].unsqueeze(0)
    image_feature_list.append(image_feature)

for j in range(N):
    input_id = input_ids[j].unsqueeze(0)
    input_id_list.append(input_id)
    label = labels[j].unsqueeze(0)
    label_list.append(label)

scores = clip_flant5_score(image_features=image_feature_list, input_ids=input_id_list, labels=label_list) # scores[i][j] is the score between image i and text j
print(scores)