import json
import cv2
import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import subprocess
import argparse
from custom_clip_vision_encoder import CustomCLIPVisionEncoder  # ✅ Custom Vision Tower 사용


# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate text-video similarity.")
parser.add_argument("--dataset", type=str, required=True, default='charades-sta', help="dataset")
parser.add_argument("--start_idx", type=int, required=False, default=0, help="video key start index")
parser.add_argument("--end_idx", type=int, required=False, default=None, help="video key end index")
args = parser.parse_args()

# JSON 파일 불러오기 (예: videos.json)
with open(f'llm_outputs_{args.dataset}.json', 'r') as f:
    video_json = json.load(f)

# 비디오 파일이 있는 두 경로 설정
if args.dataset == 'charades-sta':
    data_path = ['../PRVR/video_data/Charades_v1/']
    target_fps = 3
elif args.dataset == 'activitynet':
    data_path = ['../PRVR/video_data/Activitynet_1-2/', '../PRVR/video_data/Activitynet_1-3/']
    target_fps = 2

# 피쳐 저장 경로 설정
save_path = os.path.join('../PRVR/video_features', args.dataset, f'CLIP-L-14-336')

# 각 경로에서 파일 목록과 경로 정보를 함께 저장
all_video_files = []
for path in data_path:
    for file in os.listdir(path):
        all_video_files.append((file, path))  # (파일명, 경로) 튜플

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_tower = CustomCLIPVisionEncoder(model_name="openai/clip-vit-large-patch14-336", 
                                    select_layer=-2, select_feature="patch").to(device)
# ✅ Image Processor 가져오기
image_processor = vision_tower.image_processor

# JSON의 키 값(비디오 파일 접두어) 리스트
video_keys = list(video_json.keys())
if args.end_idx is not None:
    video_keys = video_keys[args.start_idx:args.end_idx]

# ✅ `expand2square()` 함수 추가 (이미지 크기 맞추기)
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


for key in tqdm(video_keys, desc="Processing video files", unit="file"):
    # 두 경로에서 해당 키로 시작하는 파일 검색
    matched_files = [(f, path) for (f, path) in all_video_files if f.startswith(key)]
    if not matched_files:
        print(f"No file starting with {key} found in any directory.")
        continue

    # 동일 키에 해당하는 모든 파일 처리
    for video_file, video_dir in matched_files:
        video_path = os.path.join(video_dir, video_file)
        video_feature_save_path = os.path.join(save_path, os.path.splitext(video_file)[0] + ".pth")
        
        # 이미 피쳐가 추출되어 저장된 경우 건너뛰기
        if os.path.exists(video_feature_save_path):
            print(f"Features for {video_file} already exist. Skipping.")
            continue
        
        # OpenCV로 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {video_file}")
            continue
        
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_interval = source_fps / target_fps
        
        frame_idx = 0
        next_frame = 0
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 만약 파일 끝에 도달하지 않은 상태라면(오류나 손상으로 인한 실패)
                if frame_idx < total_frames - 1:
                    # 현재 프레임 시간(초 단위)
                    time_sec = frame_idx / source_fps
                    cmd = [
                        'ffmpeg',
                        '-ss', str(time_sec),
                        '-i', video_path,
                        '-frames:v', '1',
                        '-f', 'image2pipe',
                        '-pix_fmt', 'bgr24',
                        '-vcodec', 'rawvideo',
                        '-'
                    ]
                    try:
                        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        raw_frame = pipe.stdout.read()
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frame = np.frombuffer(raw_frame, dtype='uint8').reshape((height, width, 3))
                        ret = True
                    except Exception as e:
                        print("ffmpeg fallback failed:", e)
                        break
                else:
                    # 비디오 파일의 끝에 도달한 경우
                    break

            if frame_idx == int(round(next_frame)):
                next_frame += fps_interval
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # ✅ CustomCLIPVisionEncoder의 image_processor로 이미지 전처리
                background_color = [int(c * 255) for c in image_processor.image_mean]  
                pil_image = expand2square(pil_image, tuple(background_color))
                inputs = image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"].squeeze(0).to(device)

                frames.append(inputs)  # OpenCV 프레임 (np.ndarray) 저장

            frame_idx += 1
        
        if len(frames) == 0:
            print(f"No valid frames extracted from {video_file}")
            continue

        with torch.no_grad():
            MAX_FRAME_SIZE = 128
            if len(frames) > MAX_FRAME_SIZE:
                total_image_features = []
                for i in range(0, len(frames), MAX_FRAME_SIZE):
                    batch_frames = torch.stack(frames[i:i + MAX_FRAME_SIZE], dim=0)  # ✅ 배치 구성
                    batch_features = vision_tower(batch_frames)
                    total_image_features.append(batch_features.cpu())
                total_image_features = torch.cat(total_image_features, dim=0)
            else:
                frames = torch.stack(frames, dim=0)  # ✅ 한 번에 처리 가능하면 batch 구성
                total_image_features = vision_tower(frames)

        torch.save(total_image_features.cpu(), video_feature_save_path)