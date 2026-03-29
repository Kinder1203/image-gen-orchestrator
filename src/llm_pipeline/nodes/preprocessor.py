from ..core.schemas import AgentState
from loguru import logger
import io
import os
import requests
from PIL import Image

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg library is not installed. Background removal will run in dummy mode.")

def process_uploaded_image(state: AgentState) -> dict:
    """
    (Step 3 - Image Branch) 
    사용자가 업로드한 (다중) 이미지 배열을 순회하며 진짜(Real) 누끼 처리를 합니다.
    rembg를 이용해 객체를 분리하고 순백색(Solid White) 컨버스에 이식합니다.
    """
    # 텍스트 단일 이미지라면 1개짜리 배열화, 여러 개면 모두 순회
    raw_image_url = state.get("user_image", "")
    input_urls = [raw_image_url] if raw_image_url else []
    
    logger.info(f"Rembg Multi-view Preprocessing started for {len(input_urls)} images.")
    processed_urls = []
    
    for idx, url in enumerate(input_urls):
        if not url:
            continue
            
        logger.debug(f"Processng image [{idx}]: {url}")
        
        if REMBG_AVAILABLE and url.startswith("http"):
            try:
                # 1. 이미지 다운로드 
                response = requests.get(url, timeout=10)
                input_img = Image.open(io.BytesIO(response.content)).convert("RGBA")
                
                # 2. rembg 백그라운드 제거 (투명화)
                no_bg_img = remove(input_img)
                
                # 3. 투명 부분을 흰색으로 강제 채우기 (TRELLIS 필수)
                white_bg = Image.new("RGBA", no_bg_img.size, "WHITE")
                white_bg.paste(no_bg_img, (0, 0), no_bg_img)
                final_rgb_img = white_bg.convert("RGB")
                
                # 4. 결과물 저장 후 경로 반환
                save_path = f"/tmp/processed_shoe_{idx}.png"
                os.makedirs("/tmp", exist_ok=True)
                final_rgb_img.save(save_path)
                
                processed_urls.append(save_path)
                logger.success(f"Successfully removed background and saved: {save_path}")
            except Exception as e:
                logger.error(f"Rembg processing failed for {url}: {e}")
                processed_urls.append(url) # 실패시 원본 유지
        else:
            # 더미 다운그레이드
            logger.info("Using Dummy Rembg preprocessing (No library or local file).")
            processed_urls.append(f"processed_white_bg_{idx}.png")
            
    return {
        "current_image_urls": processed_urls
    }
