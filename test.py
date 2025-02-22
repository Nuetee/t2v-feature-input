import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

### For a single (image, text) pair
image = "images/0.png" # an image path in string format
text = "someone talks on the phone angrily while another person sits happily"
score = clip_flant5_score(images=[image], texts=[text])
print(score)
### Alternatively, if you want to calculate the pairwise similarity scores 
### between M images and N texts, run the following to return a M x N score tensor.
images = ["images/0.png", "images/1.png"]
texts = ["someone talks on the phone angrily while another person sits happily",
         "someone talks on the phone happily while another person sits angrily"]
scores = clip_flant5_score(images=images, texts=texts) # scores[i][j] is the score between image i and text j
print(scores)