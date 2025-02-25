from abc import abstractmethod
from typing import List, TypedDict, Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR

class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR,
                 **kwargs):
        """Initialize the ScoreModel
        """
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str,
                           **kwargs):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                # images: Union[str, List[str]],
                # texts: Union[str, List[str]],
                image_features: Union[torch.Tensor, List[torch.Tensor]] = None,  # NEW: 직접 feature 입력 가능
                # text_features: Union[torch.Tensor, List[torch.Tensor]] = None,  # NEW: 직접 토큰 입력 가능
                input_ids: Union[torch.Tensor, List[torch.Tensor]] = None,  # NEW: 직접 토큰 입력 가능
                labels: Union[torch.Tensor, List[torch.Tensor]] = None,  # NEW: 직접 토큰 입력 가능
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        if type(image_features) == torch.Tensor:
            image_features = [image_features]
        if type(input_ids) == torch.Tensor:
            input_ids = [input_ids]
        if type(labels) == torch.Tensor:
            labels = [labels]

        scores = torch.zeros(len(image_features), len(input_ids)).to(self.device)

        for i, image in enumerate(image_features):
            scores[i] = self.model.forward([image] * len(input_ids), input_ids, labels, **kwargs)
        return scores
        # M = len(images)
        # N = len(texts)

        # # ✅ 기존 방식: for문을 돌면서 하나씩 처리 ❌
        # # ✅ 개선된 방식: 한 번에 M × N 형태로 변환하여 batch 연산 ⏩
        # scores = self.model.forward(images=images, texts=texts, **kwargs)
        # return scores
    
    def batch_forward(self,
                      dataset: List[ImageTextDict],
                      batch_size: int=16,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        num_samples = len(dataset)
        num_images = len(dataset[0]['images'])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_images, num_texts).to(self.device)
        
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = len(batch['images'][0])
            assert len(batch['images']) == num_images, \
                f"Number of image options in batch {batch_idx} is {len(batch['images'])}. Expected {num_images} images."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            for image_idx in range(num_images):
                images = batch['images'][image_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    scores[counter:counter+cur_batch_size, image_idx, text_idx] = \
                        self.model.forward(images, texts, **kwargs)
            
            counter += cur_batch_size
        return scores
    