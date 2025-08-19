"""
이미지 임베딩 생성 시스템 (OpenCLIP ViT-g-14 버전)
================================================

목적:
    train 폴더의 이미지들을 OpenCLIP ViT-g-14 모델로 임베딩하여 벡터화하고, 
    numpy 배열 형태로 저장하여 빠른 유사도 검색을 위한 인덱스를 구축합니다.

입력 스펙:
---------
1. IMAGE_DIR: "train" - 임베딩을 생성할 이미지들이 있는 폴더
2. MODEL_NAME: 'ViT-g-14' - 사용할 OpenCLIP 모델
3. BATCH_SIZE: 8 - 한 번에 처리할 이미지 개수 (GPU 메모리 효율성, g-14는 메모리 사용량이 큼)
4. RECURSIVE_SCAN: True - 하위 폴더까지 스캔할지 여부
5. ALLOWED_EXTS: {".jpg", ".jpeg", ".png", ".webp", ".bmp"} - 지원 이미지 확장자

출력 스펙:
---------
1. OUT_FEATURES: "embeddings/train_features_vit_g_14.npy" - 이미지 임베딩 벡터들 (N x 1024)
2. OUT_PATHS: "embeddings/train_paths_vit_g_14.npy" - 이미지 파일 경로들 (N개)

사용법:
------
python embeddings_great.py

주의사항:
---------
- GPU 메모리가 충분해야 합니다 (RTX 4090 권장, BATCH_SIZE 조정 필요시)
- train 폴더에 이미지 파일들이 있어야 합니다
- 첫 실행 시 OpenCLIP 모델 다운로드가 필요합니다
- open_clip_torch 라이브러리가 설치되어 있어야 합니다
"""

import os
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
import torch
import open_clip
from typing import Callable, Tuple

# ==================== 시스템 설정 ====================
IMAGE_DIR: str = "train"                    # 임베딩을 생성할 이미지 폴더
OUT_FEATURES: str = "embeddings/train_features_vit_g_14.npy"  # 임베딩 벡터 저장 파일
OUT_PATHS: str = "embeddings/train_paths_vit_g_14.npy"        # 이미지 경로 저장 파일

# OpenCLIP 모델 설정
MODEL_NAME: str = 'ViT-g-14'      # 거대 모델 (1024차원, 매우 정확함)     
 # 사전 훈련된 가중치
PRETRAINED = "laion2b_s34b_b88k"


BATCH_SIZE: int = 8                         # 배치 크기 (GPU 메모리에 따라 조정, g-14는 메모리 사용량이 큼)
RECURSIVE_SCAN: bool = True                 # 하위 폴더까지 스캔할지 여부
ALLOWED_EXTS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # 지원 이미지 확장자

# ==================== 핵심 함수들 ====================

def list_images(root: str, recursive: bool = True) -> list[str]:
    """
    지정된 폴더에서 이미지 파일들의 경로를 수집합니다.
    
    입력:
        root (str): 이미지를 찾을 루트 폴더 경로
        recursive (bool): 하위 폴더까지 스캔할지 여부
    
    출력:
        list[str]: 이미지 파일들의 절대 경로 리스트 (정렬됨)
    
    예시:
        >>> list_images("used", recursive=False)
        ['used/img1.jpg', 'used/img2.png', ...]
    """
    paths: list[str] = []
    
    if recursive:
        # 하위 폴더까지 재귀적으로 스캔
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
                    full_path: str = os.path.join(dirpath, filename)
                    paths.append(full_path)
    else:
        # 현재 폴더만 스캔
        for filename in os.listdir(root):
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTS:
                full_path: str = os.path.join(root, filename)
                paths.append(full_path)
    
    # 파일명 순으로 정렬하여 일관된 결과 보장
    paths.sort()
    return paths


def load_model(model_name: str, pretrained: str, device: torch.device) -> Tuple[torch.nn.Module, Callable]:
    """
    OpenCLIP 모델과 전처리 변환을 로드합니다.
    FP16 (Half Precision) + TorchScript JIT 컴파일을 적용하여 메모리 사용량을 줄이고 속도를 향상시킵니다.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()
    
    # FP16으로 변환하여 메모리 사용량 감소 및 속도 향상
    if device.type == 'cuda':
        model.half()
        print("✅ FP16 (Half Precision) 적용 완료 - 메모리 사용량 50% 감소, 속도 향상")
        
        # TorchScript JIT 컴파일로 추가 속도 향상
        try:
            # 더미 입력으로 JIT 컴파일 (FP16 모델에 맞춤)
            dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
            model = torch.jit.trace(model, dummy_input)
            print("✅ TorchScript JIT 컴파일 완료 - 추가 10-20% 속도 향상")
        except Exception as e:
            print(f"⚠️  TorchScript 컴파일 실패, 원본 모델 사용: {e}")
    
    return model, preprocess


def embed_batch(model: torch.nn.Module, transforms_obj: Callable, 
                pil_images: list[Image.Image], device: torch.device) -> np.ndarray:
    """
    이미지 배치를 OpenCLIP 임베딩 벡터로 변환합니다.
    FP16 최적화가 적용되어 있습니다.
    
    입력:
        model (open_clip.CLIP): 로드된 OpenCLIP 모델
        transforms_obj (open_clip.transform.Transforms): OpenCLIP 전처리 변환
        pil_images (list[Image.Image]): PIL Image 객체들의 리스트
        device (torch.device): GPU 또는 CPU 디바이스
    
    출력:
        np.ndarray: 정규화된 임베딩 벡터들 (배치크기 x 차원수)
                   - ViT-g-14: 1024차원
    
    예시:
        >>> images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
        >>> embeddings = embed_batch(model, transforms_obj, images, device)
        >>> print(embeddings.shape)  # (2, 1024) for ViT-g-14
    """
    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
        # 이미지들을 OpenCLIP 모델 입력 형식으로 변환
        inputs = torch.stack([transforms_obj(img) for img in pil_images])
        
        # FP16으로 변환하여 메모리 사용량 감소
        if device.type == 'cuda':
            inputs = inputs.half()
        
        inputs = inputs.to(device)
        
        # OpenCLIP 모델로 이미지 특징 추출
        feats: torch.Tensor = model.encode_image(inputs)
        
        # L2 정규화 (코사인 유사도 계산을 위해)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    
    # detach()를 추가하여 그래디언트 계산 그래프에서 분리, 메모리 사용량 감소
    # GPU에서 CPU로 이동하고 numpy 배열로 변환
    return feats.detach().cpu().numpy().astype("float32")

def main() -> None:
    """
    메인 실행 함수: train 폴더의 모든 이미지를 OpenCLIP 임베딩하여 파일로 저장합니다.
    
    입력: 없음
    
    출력: 없음 (파일로 저장됨)
    
    처리 과정:
    1. train 폴더 존재 확인
    2. GPU/CPU 디바이스 선택
    3. OpenCLIP 모델 로드
    4. 이미지 파일 경로 수집
    5. 배치 단위로 임베딩 생성
    6. 결과를 numpy 배열로 저장
    
    예외 처리:
    - FileNotFoundError: IMAGE_DIR이 존재하지 않을 때
    - RuntimeError: 이미지 파일을 찾을 수 없을 때
    - UnidentifiedImageError: 손상된 이미지 파일
    - OSError: 파일 읽기 오류
    """
    start_time: float = time.time()
    
    # 1. train 폴더 존재 확인
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    
    # 2. GPU/CPU 디바이스 선택
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # 3. OpenCLIP 모델 로드
    print("🔧 OpenCLIP 모델 로드 중...")
    print(f"   모델: {MODEL_NAME}")
    print(f"   가중치: {PRETRAINED}")
    model, transforms = load_model(MODEL_NAME, PRETRAINED, device)
    print("✅ 모델 로드 완료")
    
    # 4. 이미지 파일 경로 수집
    print("🔍 이미지 파일 스캔 중...")
    img_paths: list[str] = list_images(IMAGE_DIR, recursive=RECURSIVE_SCAN)
    
    if not img_paths:
        raise RuntimeError(f"No images found under: {IMAGE_DIR}")
    
    print(f"📊 총 {len(img_paths)}개의 이미지 파일 발견")
    
    # 5. 배치 단위로 임베딩 생성
    features: list[np.ndarray] = []        # 임베딩 벡터들을 저장할 리스트
    kept_paths: list[str] = []             # 성공적으로 처리된 이미지 경로들
    batch: list[Image.Image] = []          # 현재 배치의 이미지들
    batch_paths: list[str] = []            # 현재 배치의 경로들
    
    print(f"⚡ 임베딩 생성 시작 (batch_size={BATCH_SIZE}, device={device})")
    
    # 진행률 표시와 함께 이미지 처리
    for img_path in tqdm(img_paths, desc="이미지 처리"):
        try:
            # 이미지 로드 및 RGB 변환
            img: Image.Image = Image.open(img_path).convert("RGB")
            batch.append(img)
            batch_paths.append(img_path)
            
            # 배치가 가득 찼을 때 임베딩 생성
            if len(batch) == BATCH_SIZE:
                feats: np.ndarray = embed_batch(model, transforms, batch, device)
                features.append(feats)
                kept_paths.extend(batch_paths)
                
                # 배치 초기화
                batch, batch_paths = [], []
                
        except (UnidentifiedImageError, OSError) as e:
            # 손상된 이미지나 읽기 오류 시 건너뛰기
            print(f"⚠️  [SKIP] 로드 실패: {img_path} ({e})")
    
    # 마지막 배치 처리 (BATCH_SIZE보다 적은 경우)
    if batch:
        feats: np.ndarray = embed_batch(model, transforms, batch, device)
        features.append(feats)
        kept_paths.extend(batch_paths)
    
    # 6. 모든 임베딩을 하나의 배열로 결합
    if features:
        feats_all: np.ndarray = np.vstack(features)
    else:
        # 이미지가 없는 경우 빈 배열 생성 (ViT-g-14는 1024차원)
        feats_all: np.ndarray = np.zeros((0, 1024), dtype="float32")
    
    # 7. 결과를 numpy 배열로 저장
    print("💾 임베딩 저장 중...")
    np.save(OUT_FEATURES, feats_all)
    np.save(OUT_PATHS, np.array(kept_paths, dtype=object))
    
    # 8. 완료 정보 출력
    elapsed_time: float = time.time() - start_time
    print("✅ 인덱스 저장 완료")
    print(f" 📊 features: {OUT_FEATURES} {feats_all.shape}")
    print(f" 📁 paths   : {OUT_PATHS} {len(kept_paths)}개")
    print(f" ⏱️  경과시간: {elapsed_time:.1f}초")
    
    # 9. 성공률 계산
    success_rate: float = (len(kept_paths) / len(img_paths)) * 100
    print(f" 🎯 성공률: {success_rate:.1f}% ({len(kept_paths)}/{len(img_paths)})")
    
    # 10. 모델 정보 출력
    print(f" 🚀 모델 정보:")
    print(f"    - 모델명: {MODEL_NAME}")
    print(f"    - 가중치: {PRETRAINED}")
    print(f"    - 임베딩 차원: {feats_all.shape[1] if feats_all.shape[0] > 0 else 1024}")
    print(f"    - 배치 크기: {BATCH_SIZE}")

# ==================== 스크립트 실행 ====================

if __name__ == "__main__":
    try:
        main()
        print("\nOpenCLIP ViT-g-14 임베딩 생성이 성공적으로 완료되었습니다!")
        print("이제 train 폴더의 이미지들이 1024차원 고품질 임베딩으로 벡터화되었습니다.")
        print("\n사용법:")
        print("1. predict.py에서 FEATURES_NPY와 PATHS_NPY를 이 파일들의 경로로 변경")
        print("2. MODEL_NAME을 'open_clip/ViT-g-14'로 설정")
        print("3. python predict.py 실행")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   다음 사항들을 확인해주세요:")
        print("   1. train 폴더가 존재하는지")
        print("   2. train 폴더에 이미지 파일이 있는지")
        print("   3. GPU 메모리가 충분한지 (RTX 4090 권장, BATCH_SIZE 조정 필요시)")
        print("   4. 인터넷 연결이 안정적인지 (모델 다운로드용)")
        print("   5. open_clip_torch 라이브러리가 설치되어 있는지")
        print("      설치 명령어: pip install open_clip_torch")
