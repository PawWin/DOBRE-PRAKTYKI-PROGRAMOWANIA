import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR


class PlateOCR:
    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "en",
        ocr_version: str = "PP-OCRv4",
        text_det_thresh: float = 0.05,
        text_det_box_thresh: float = 0.05,
        text_rec_score_thresh: float = 0.0,
        top_k: int = 4,
    ):
        if use_gpu and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(f"PlateOCR: Using GPU | version={ocr_version}")
        else:
            paddle.device.set_device("cpu")
            print(f"PlateOCR: Using CPU | version={ocr_version}")
        
        self.ocr = PaddleOCR(
            lang=lang,
            ocr_version=ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_rec_score_thresh=text_rec_score_thresh,
        )
        
        self.top_k = top_k
    
    def recognize(self, image: np.ndarray, preprocess: bool = False) -> str:
        if image is None or image.size == 0:
            return ""
        if preprocess:
            image = self._preprocess(image)
        result = self.ocr.predict(image)
        return self._first_text(result)
    
    def recognize_batch(self, images: list[np.ndarray]) -> list[str]:
        results = []
        for image in images:
            text = self.recognize(image)
            results.append(text)
        return results

    def recognize_list(self, images: list[np.ndarray]) -> list[str]:
        if not images:
            return []
        result = self.ocr.predict(images)
        if not result:
            return ["" for _ in images]
        texts = []
        for item in result:
            texts.append(self._first_text([item]))
        while len(texts) < len(images):
            texts.append("")
        return texts
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h < 50 or w < 100:
            scale = max(50 / h, 100 / w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _first_text(self, result: list) -> str:
        if not result:
            return ""
        item = result[0]
        if isinstance(item, dict):
            for text in item.get("rec_texts", []):
                cleaned = text.strip()
                if cleaned:
                    return cleaned
            return ""
        if isinstance(item, (list, tuple)):
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_data = line[1]
                    if isinstance(text_data, tuple) and len(text_data) >= 2:
                        texts.append(text_data[0])
                    elif isinstance(text_data, str):
                        texts.append(text_data)
            return "".join(texts)
        return ""