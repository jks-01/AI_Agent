"""
이미지 유사도 검색 시스템 (OpenCLIP ViT-g-14 버전)
================================================

입력 스펙:
---------
1. INPUT_DIR: "test" 폴더에 검색할 이미지 파일들 (.jpg, .jpeg, .png, .webp, .bmp)
2. USED_DIR: "train" 폴더에 비교 대상 이미지들
3. FEATURES_NPY: "embeddings/train_features_vit_g_14.npy" - train 폴더 이미지들의 OpenCLIP 임베딩 벡터
4. PATHS_NPY: "embeddings/train_paths_vit_g_14.npy" - train 폴더 이미지들의 파일 경로
5. PRODUCTS_CSV: "records/carbot_data_final.csv" - 상품 정보 (상품명, 정가, 중고가, 링크)
6. MODEL_NAME: 'open_clip/ViT-g-14' - OpenCLIP 모델명
7. TOPK: 5 - 검색 결과 상위 개수

출력 스펙:
---------
1. 시각화: 쿼리 이미지 + top k 유사 이미지들을 1행으로 표시
2. 텍스트: 각 이미지의 순위, 파일명, 폴더명, 유사도, 정가, 중고가, 링크
3. 반환값: 검색 결과 딕셔너리 리스트 (return_results=True일 때)

사용법:
------
search_by_image_name("이미지파일명.jpg")  # 시각화 포함
get_search_results_only("이미지파일명.jpg")  # 결과만 반환
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 라이브러리 충돌 방지
import gc
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
warnings.filterwarnings('ignore')
import time

from PIL import Image
import torch
import open_clip

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 시스템 설정 ====================
INPUT_DIR: str = "test"                     # 검색할 이미지가 있는 폴더 (test 폴더)
USED_DIR: str = "train"                     # 비교 대상 이미지들이 있는 폴더 (train 폴더)

FEATURES_NPY: str = "embeddings/train_features_vit_g_14.npy"  # train 폴더 이미지들의 임베딩 벡터
PATHS_NPY: str = "embeddings/train_paths_vit_g_14.npy"        # train 폴더 이미지들의 파일 경로
MODEL_NAME: str = 'ViT-g-14'     # OpenCLIP 모델명 (정확한 모델명으로 수정 필요)

PRODUCTS_CSV: str = "records/carbot_data_final.csv"     # 상품 정보 CSV 파일

TOPK: int = 1                              # 검색 결과 상위 개수

def load_model(model_name: str, device: torch.device) -> Tuple[open_clip.CLIP, Callable]:
    """
    OpenCLIP 모델과 전처리 함수를 로드합니다.
    FP16 (Half Precision)을 적용하여 메모리 사용량을 줄이고 속도를 향상시킵니다.
    """
    print(f"🔍 모델 로드 시작: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    model.eval()
    
    # FP16으로 변환하여 메모리 사용량 감소 및 속도 향상
    if device.type == 'cuda':
        model.half()
        # print("✅ FP16 (Half Precision) 적용 완료 - 메모리 사용량 50% 감소, 속도 향상")
    
    return model, preprocess

def embed_image(model: open_clip.CLIP, transforms_obj: Callable, img_path: str, device: torch.device) -> np.ndarray:
    """
    이미지의 OpenCLIP 임베딩 벡터를 계산합니다.
    FP16 최적화가 적용되어 있습니다.
    """
    from PIL import Image
    import torch
    import numpy as np

    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        # FP16으로 변환하여 메모리 사용량 감소
        inputs = transforms_obj(image).unsqueeze(0)
        if device.type == 'cuda':
            inputs = inputs.half()
        inputs = inputs.to(device)
        
        feat = model.encode_image(inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    
    # detach()를 추가하여 그래디언트 계산 그래프에서 분리, 메모리 사용량 감소
    return feat.detach().cpu().numpy().astype("float32").flatten()

def list_input_images() -> List[str]:
    """
    input 폴더의 이미지 파일들을 리스트로 반환합니다.
    
    입력: 없음
    
    출력:
        List[str]: 지원되는 이미지 확장자를 가진 파일명 리스트
    """
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ {INPUT_DIR} 폴더를 찾을 수 없습니다.")
        return []
    
    allowed_exts: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images: List[str] = []
    
    for f in os.listdir(INPUT_DIR):
        if os.path.splitext(f)[1].lower() in allowed_exts:
            images.append(f)
    
    images.sort()
    return images

def load_products_info() -> Dict[str, Dict[str, Any]]:
    """
    carbot_data_final.csv 파일을 로드하여 상품 정보를 반환합니다.
    
    입력: 없음
    
    출력:
        Dict[str, Dict[str, Any]]: {상품명: {retail_price, used_price_avg, retail_link}} 형태의 딕셔너리
    """
    try:
        if os.path.isfile(PRODUCTS_CSV):
            df: pd.DataFrame = pd.read_csv(PRODUCTS_CSV)
            products_dict: Dict[str, Dict[str, Any]] = {}
            
            for _, row in df.iterrows():
                product_name: str = str(row['product_name']).strip()
                retail_price: Any = row['retail_price']
                used_price_avg: Any = row['used_price_avg']
                retail_link: str = str(row['retail_link']).strip()
                
                products_dict[product_name] = {
                    'retail_price': retail_price,
                    'used_price_avg': used_price_avg,
                    'retail_link': retail_link
                }
            
            print(f"✅ {len(products_dict)}개의 상품 정보를 로드했습니다.")
            return products_dict
        else:
            print(f"⚠️  {PRODUCTS_CSV} 파일을 찾을 수 없습니다.")
            return {}
    except Exception as e:
        print(f"❌ 상품 정보 로드 실패: {e}")
        return {}

def get_product_info_from_path(image_path: str, products_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    이미지 경로에서 폴더명을 추출하여 해당 상품의 정보를 반환합니다.
    
    입력:
        image_path (str): 이미지 파일 경로
        products_dict (Dict[str, Dict[str, Any]]): 상품 정보 딕셔너리
    
    출력:
        Dict[str, Any]: 상품 정보 또는 기본값
    """
    # 경로에서 폴더명 추출 (마지막 폴더명이 상품명)
    path_parts = image_path.replace('\\', '/').split('/')
    folder_name = path_parts[-2] if len(path_parts) > 1 else ""
    
    # 폴더명에서 상품명 추출 (헬로카봇_ 접두사 제거)
    if folder_name.startswith("헬로카봇_"):
        product_name = folder_name[6:]  # "헬로카봇_" 제거
    else:
        product_name = folder_name
    
    # 정확한 매칭 시도
    if product_name in products_dict:
        return products_dict[product_name]
    
    # 부분 매칭 시도 (폴더명이 상품명의 일부인 경우)
    for key, info in products_dict.items():
        if product_name in key or key in product_name:
            return info
    
    # 매칭되지 않는 경우 기본값 반환
    return {
        'retail_price': '가격 정보 없음',
        'used_price_avg': '중고가 정보 없음',
        'retail_link': '링크 없음'
    }

def search_similar_images(query_image_path: str, features: np.ndarray, paths: np.ndarray, 
                         model: open_clip.CLIP, transforms_obj: Callable, device: torch.device, 
                         top_k: int = 5) -> Tuple[List[Dict[str, Union[int, float, str]]], float, float]:
    """
    이미지 유사도 검색을 수행합니다.
    
    입력:
        query_image_path (str): 검색할 이미지 경로
        features (np.ndarray): 모든 이미지의 임베딩 벡터 (N x 차원)
        paths (np.ndarray): 모든 이미지의 파일 경로
        model (open_clip.CLIP): OpenCLIP 모델 객체
        transforms_obj (open_clip.transform.Transforms): OpenCLIP 전처리 변환
        device (torch.device): GPU 또는 CPU 디바이스
        top_k (int): 반환할 상위 결과 개수
    
    출력:
        Tuple[List[Dict[str, Union[int, float, str]]], float, float]: 
            (검색 결과 딕셔너리 리스트, 임베딩 생성 시간, 유사도 계산 시간)
            [{'rank': 1, 'path': '경로', 'similarity': 0.8, 'filename': '파일명'}, ...]
    """
    try:
        # 쿼리 이미지 임베딩 계산 시간 측정
        embedding_start = time.time()
        q: np.ndarray = embed_image(model, transforms_obj, query_image_path, device)
        embedding_time = time.time() - embedding_start
        
        # 차원 확인 및 조정
        query_dim = q.shape[0]
        features_dim = features.shape[1]
        
        # print(f"🔍 차원 정보: 쿼리 이미지 {query_dim}차원, 저장된 임베딩 {features_dim}차원")
        
        if query_dim != features_dim:
            print(f"⚠️  차원 불일치 감지! 쿼리: {query_dim}차원, 저장된 임베딩: {features_dim}차원")
            print("   저장된 임베딩과 동일한 모델을 사용해야 합니다.")
            return [], embedding_time, 0.0
        
        # 유사도 계산 시간 측정
        similarity_start = time.time()
        
        # 코사인 유사도 계산을 위한 정규화
        q = q / (np.linalg.norm(q) + 1e-9)
        features_norm: np.ndarray = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
        
        # 유사도 계산 (정규화된 벡터의 내적 = 코사인 유사도)
        sims: np.ndarray = features_norm @ q
        # sims = np.dot(features_norm, q) 
        top_idx: np.ndarray = np.argsort(sims)[::-1][:top_k]  # 유사도 내림차순 정렬
        
        # 결과 생성
        results: List[Dict[str, Union[int, float, str]]] = []
        for rank, i in enumerate(top_idx, start=1):
            results.append({
                'rank': rank,
                'path': str(paths[i]),
                'similarity': float(sims[i]),
                'filename': os.path.basename(str(paths[i]))
            })
        
        similarity_time = time.time() - similarity_start
        
        return results, embedding_time, similarity_time
        
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return [], 0.0, 0.0

def visualize_search_results_with_query(query_image_path: str, results: List[Dict[str, Union[int, float, str]]], 
                                      products_dict: Dict[str, Any], figsize: Tuple[int, int] = (20, 8)) -> None:
    """
    검색 결과를 시각화합니다. (쿼리 이미지 + 결과 이미지들)
    
    입력:
        query_image_path (str): 쿼리 이미지 경로
        results (List[Dict[str, Union[int, float, str]]]): 검색 결과 리스트
        products_dict (Dict[str, Any]): 상품 정보 딕셔너리
        figsize (Tuple[int, int]): 그래프 크기 (가로, 세로)
    
    출력: 없음 (matplotlib 그래프 표시)
    """
    if not results:
        print("❌ 검색 결과가 없습니다.")
        return
    
    # 쿼리 이미지 로드
    try:
        query_img: Image.Image = Image.open(query_image_path).convert('RGB')
        print(f"✅ 쿼리 이미지 로드 완료: {os.path.basename(query_image_path)}")
    except Exception as e:
        print(f"❌ 쿼리 이미지 로드 실패: {e}")
        return
    
    # 결과 이미지들 로드
    result_images: List[Image.Image] = []
    for result in results:
        try:
            img: Image.Image = Image.open(str(result['path'])).convert('RGB')
            result_images.append(img)
        except Exception as e:
            print(f"❌ 결과 이미지 로드 실패: {result['path']} - {e}")
            # 빈 이미지로 대체
            result_images.append(Image.new('RGB', (224, 224), color='gray'))
    
    # 1행으로 시각화 (쿼리 이미지 + top k 결과)
    total_images: int = 1 + len(results)  # 쿼리 이미지 1개 + 결과 이미지들
    fig, axes = plt.subplots(1, total_images, figsize=figsize)
    fig.suptitle(f'Similarity results: {os.path.basename(query_image_path)}', fontsize=20, fontweight='bold')
    
    # 1번째 위치: 쿼리 이미지
    ax = axes[0] if total_images > 1 else axes
    ax.imshow(query_img)
    
    # 쿼리 이미지 경로에서 폴더명 추출
    query_path_parts = query_image_path.replace('\\', '/').split('/')
    query_folder_name = query_path_parts[-2] if len(query_path_parts) > 1 else "알 수 없음"
    
    ax.set_title(f'Input image\n{query_folder_name}', fontsize=14, fontweight='bold', color='blue')
    ax.axis('off')
    
    # 2번째 위치부터: 결과 이미지들
    for i, (result, img) in enumerate(zip(results, result_images)):
        ax = axes[i + 1] if total_images > 1 else axes
        
        ax.imshow(img)
        
        # 가격 정보 가져오기
        price_info: Dict[str, Any] = get_product_info_from_path(str(result['path']), products_dict)
        
        # 제목에 순위, 파일명, 유사도, 가격 표시
        retail_price = price_info['retail_price'] if price_info['retail_price'] != '가격 정보 없음' else 'N/A'
        used_price = price_info['used_price_avg'] if price_info['used_price_avg'] != '중고가 정보 없음' else 'N/A'
        
        title: str = f"#{result['rank']}\n{result['filename']}\n유사도: {result['similarity']:.4f}\n정가: {retail_price}\n중고가: {used_price}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 유사도에 따른 테두리 색상 설정
        if float(result['similarity']) > 0.8:
            color: str = 'green'      # 높은 유사도
        elif float(result['similarity']) > 0.6:
            color: str = 'orange'     # 중간 유사도
        else:
            color: str = 'red'        # 낮은 유사도
        
        # 테두리 추가
        rect = patches.Rectangle((0, 0), img.width-1, img.height-1, 
                                linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    
    # 텍스트 결과도 출력
    print(f"\n📊 검색 결과 요약 (상위 {len(results)}개)")
    print("=" * 100)
    for result in results:
        price_info: Dict[str, Any] = get_product_info_from_path(str(result['path']), products_dict)
        
        # 경로에서 폴더명과 파일명 추출
        path_parts = str(result['path']).replace('\\', '/').split('/')
        folder_name = path_parts[-2] if len(path_parts) > 1 else "알 수 없음"
        filename = result['filename']
        
        # 가격 정보 포맷팅
        retail_price = price_info['retail_price'] if price_info['retail_price'] != '가격 정보 없음' else 'N/A'
        used_price = price_info['used_price_avg'] if price_info['used_price_avg'] != '중고가 정보 없음' else 'N/A'
        
        print(f"{result['rank']:2d}. {filename}")
        print(f"    📁 폴더: {folder_name}")
        print(f"    💰 정가: {retail_price}")
        print(f"    💰 중고가: {used_price}")
        print(f"    📍 유사도: {result['similarity']:.4f}")
        print(f"    🔗 링크: {price_info['retail_link']}")
        print("-" * 80)

def search_by_image_name(image_name: str, return_results: bool = False) -> Optional[Union[List[Dict[str, Union[int, float, str]]], Dict[str, Union[str, int]]]]:
    """
    이미지명을 입력받아 유사도 검색을 수행하고 결과를 시각화합니다.
    
    입력:
        image_name (str): 검색할 이미지 파일명
        return_results (bool): True면 top1 결과를 JSON 형태로 반환, False면 시각화만
    
    출력:
        Optional[Union[List[Dict[str, Union[int, float, str]]], Dict[str, Union[str, int]]]]: 
            return_results=True일 때 top1 결과 JSON, False일 때 None
    """
    
    print(f"🔍 검색 시작: {image_name}")
    
    # 입력 이미지 경로 확인
    query_image_path: str = os.path.join(INPUT_DIR, image_name)
    if not os.path.isfile(query_image_path):
        print(f"❌ 이미지를 찾을 수 없습니다: {query_image_path}")
        print(f"\n📁 {INPUT_DIR} 폴더의 사용 가능한 이미지들:")
        input_images: List[str] = list_input_images()
        for i, img in enumerate(input_images[:20], 1):
            print(f"  {i:2d}. {img}")
        if len(input_images) > 20:
            print(f"  ... 및 {len(input_images) - 20}개 더")
        return None
    
    # 임베딩 파일 확인
    if not os.path.isfile(FEATURES_NPY) or not os.path.isfile(PATHS_NPY):
        print("❌ train 폴더의 임베딩 파일을 먼저 생성하세요.")
        print("   python embeddings_huge.py")
        return None
    
    # 상품 정보 로드 시간 측정
    products_start = time.time()
    products_dict: Dict[str, Any] = load_products_info()
    products_time = time.time() - products_start
    
    # 임베딩 파일 로드 시간 측정
    files_start = time.time()
    try:
        print("📂 임베딩 파일 로드 중...")
        feats: np.ndarray = np.load(FEATURES_NPY).astype("float32")
        paths: np.ndarray = np.load(PATHS_NPY, allow_pickle=True)
        
        if feats.ndim != 2 or feats.shape[0] != len(paths):
            raise ValueError("features/paths 파일 크기가 일치하지 않습니다.")
        
        print(f"✅ 임베딩 파일 로드 완료: {feats.shape[0]}개 벡터, {feats.shape[1]}차원")
            
    except Exception as e:
        print(f"❌ 임베딩 파일 로드 실패: {e}")
        return None
    files_time = time.time() - files_start
    
    # 모델 로드 시간 측정
    model_start = time.time()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    try:
        print("🔧 OpenCLIP 모델 로드 중...")
        print(f"   모델: {MODEL_NAME}")
        model, transforms = load_model(MODEL_NAME, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    model_time = time.time() - model_start
    
    # 유사도 검색 수행
    print(f"🔍 '{image_name}' 이미지로 유사도 검색 중... (상위 {TOPK}개 결과)")
    results, embedding_time, similarity_time = search_similar_images(query_image_path, feats, paths, model, transforms, device, TOPK)
    
    if results:
        print(f"✅ 검색 완료: {len(results)}개 결과 발견")
        
        # 시간 분석 출력
        print(f"\n⏱️  상세 시간 분석:")
        print(f"   📊 임베딩 생성 시간: {embedding_time:.3f}초")
        print(f"   🔍 유사도 계산 시간: {similarity_time:.3f}초")
        print(f"   🤖 모델 로드 시간: {model_time:.3f}초")
        print(f"   📁 파일 로드 시간: {files_time:.3f}초")
        print(f"   💰 상품 정보 로드 시간: {products_time:.3f}초")
        
        # 가격 정보를 결과에 추가
        for result in results:
            result['price'] = get_product_info_from_path(str(result['path']), products_dict)['retail_price']
        
        if return_results:
            # top1 결과만 JSON 형태로 리턴
            top1_result = results[0]  # 가장 유사한 이미지 (rank=1)
            price_info: Dict[str, Any] = get_product_info_from_path(str(top1_result['path']), products_dict)
            
            # 경로에서 폴더명 추출
            path_parts = str(top1_result['path']).replace('\\', '/').split('/')
            folder_name = path_parts[-2] if len(path_parts) > 1 else "알 수 없음"
            
            # JSON 형태로 top1 결과 구성
            top1_json = {
                "similar_toy_name": folder_name,
                "similar_image_path": str(top1_result['path']),
                "similar_retail_price": price_info['retail_price'] if price_info['retail_price'] != '가격 정보 없음' else 0,
                "similar_used_price": int(price_info['used_price_avg']) if price_info['used_price_avg'] != '중고가 정보 없음' else 0
            }
            
            return top1_json
        else:
            # 시각화 포함
            visualize_search_results_with_query(query_image_path, results, products_dict)
            return None
    else:
        print("❌ 검색 결과가 없습니다.")
        return None

# ==================== 유틸리티 함수들 ====================

def get_search_results_only(image_name: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    이미지 검색 결과만 반환하는 함수 (시각화 제외)
    
    입력:
        image_name (str): 검색할 이미지 파일명
    
    출력:
        Optional[Dict[str, Union[str, int]]]: top1 검색 결과 JSON
    """
    print(f"🔍 {image_name} 검색 시작 (결과만 반환)")
    return search_by_image_name(image_name, return_results=True)

def test_search() -> Optional[Dict[str, Union[str, int]]]:
    """
    검색 기능을 테스트합니다.
    
    입력: 없음
    
    출력:
        Optional[Dict[str, Union[str, int]]]: 테스트 결과 JSON
    """
    print("🧪 검색 기능 테스트 시작")
    
    # input 폴더의 이미지 확인
    input_images: List[str] = list_input_images()
    if not input_images:
        print("❌ input 폴더에 이미지가 없습니다.")
        return None
    
    print(f"📁 input 폴더에서 {len(input_images)}개의 이미지를 찾았습니다.")
    
    # 첫 번째 이미지로 테스트
    test_image: str = input_images[0]
    print(f"🔍 테스트 이미지: {test_image}")
    
    # 검색 실행
    results: Optional[Dict[str, Union[str, int]]] = get_search_results_only(test_image)
    
    if results:
        print(f"✅ 테스트 성공! Top1 결과 JSON 반환")
        return results
    else:
        print("❌ 테스트 실패")
        return None

def show_image(image_path: str, title: Optional[str] = None, figsize: Tuple[int, int] = (5, 3)) -> None:
    """
    이미지를 표시하는 함수
    
    입력:
        image_path (str): 이미지 파일 경로
        title (Optional[str]): 이미지 제목
        figsize (Tuple[int, int]): 그래프 크기
    
    출력: 없음 (matplotlib 그래프 표시)
    """
    try:
        img: Image.Image = Image.open(image_path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        else:
            plt.title(image_path)
        plt.show()
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")

def clear_gpu_memory() -> None:
    """
    GPU 메모리를 정리합니다.
    
    입력: 없음
    
    출력: 없음
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU 메모리 정리 완료")
    
    gc.collect()
    print("🧹 시스템 메모리 정리 완료")

def check_gpu_memory() -> Tuple[float, float, float]:
    """
    GPU 메모리 사용량을 확인합니다.
    
    입력: 없음
    
    출력:
        Tuple[float, float, float]: (할당된 메모리, 예약된 메모리, 총 메모리) (GB 단위)
    """
    if torch.cuda.is_available():
        allocated: float = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved: float = torch.cuda.memory_reserved() / 1024**3    # GB
        total: float = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # print(f"🖥️  GPU 메모리 상태:")
        # print(f"   할당됨: {allocated:.2f} GB")
        # print(f"   예약됨: {reserved:.2f} GB")
        # print(f"   총 용량: {total:.2f} GB")
        # print(f"   사용 가능: {total - reserved:.2f} GB")
        
        return allocated, reserved, total
    else:
        print("💻 GPU를 사용할 수 없습니다.")
        return 0.0, 0.0, 0.0

# ==================== 메인 실행 부분 ====================

if __name__ == "__main__":
    # check_gpu_memory()
    # clear_gpu_memory()
    
    # 예시 검색 실행 (test 폴더의 이미지로 train 폴더 검색)
    t0 = time.time()
    print("🚀 OpenCLIP ViT-g-14 모델을 사용한 이미지 유사도 검색 시작")
    # print(f"📁 검색 대상: {INPUT_DIR} 폴더")
    # print(f"📊 비교 대상: {USED_DIR} 폴더 (OpenCLIP 임베딩)")
    # print(f"🔧 모델: {MODEL_NAME}")
    # print(f"📏 임베딩 차원: 1024차원")
    print("=" * 80)
    
    # 에이전트에 넘겨줄 리턴값
    result = search_by_image_name("헬로카봇_스피너블/thunder_0150.webp", return_results=True)
    print("=" * 80)
    print("🔍 검색 결과 JSON:")
    print(result)
    print("=" * 80)
    
    total_time = time.time() - t0
    
    print(f"🎯 총 소요 시간: {total_time:.3f}초")
    print("=" * 80)
    
    # clear_gpu_memory()
    exit()