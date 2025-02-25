import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class CustomCLIPVisionEncoder(nn.Module):
    """
    Vision Tower에서 사용하는 방식과 동일하게 CLIP 모델을 로드하고 특정 hidden state를 선택하는 클래스.
    """
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", select_layer=-2, select_feature="patch"):
        super().__init__()

        # Pretrained CLIP Vision Model 로드
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name)
        self.vision_tower.requires_grad_(False)  # 학습되지 않도록 Freeze

        # Feature 선택 관련 설정
        self.select_layer = select_layer
        self.select_feature = select_feature

    def feature_select(self, image_forward_outs):
        """ 특정 hidden state를 선택하여 Feature를 가져옴 """
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            # CLS 토큰을 제외하고 패치 임베딩만 선택
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            # CLS 토큰 포함
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        return image_features

    def forward(self, images):
        """
        이미지를 CLIP 모델에 통과시키고, 특정 hidden state의 Feature를 가져온 후 반환
        """
        image_forward_outs = self.vision_tower(images, output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs)
        return image_features  # Projection Layer 없이 CLIP hidden states 반환