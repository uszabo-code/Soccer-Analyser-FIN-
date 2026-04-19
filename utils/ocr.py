"""Jersey number OCR utilities."""

from __future__ import annotations

import os
import re
import cv2
import numpy as np


def _init_paddleocr():
    """Lazy-init PaddleOCR to avoid slow import at startup."""
    try:
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        engine = PaddleOCR(
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="en_PP-OCRv4_mobile_rec",
            use_textline_orientation=False,
            lang="en",
            device="cpu",
        )

        # Detect API version: v5 uses predict(), older uses ocr()
        api_version = "v5" if hasattr(engine, 'predict') else "legacy"
        return engine, api_version
    except ImportError:
        return None, None


def _init_easyocr():
    """Fallback to EasyOCR if PaddleOCR is unavailable."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except ImportError:
        return None


_ocr_engine = None
_ocr_type = None
_paddle_api = None


def get_ocr_engine():
    """Get or initialize the OCR engine (PaddleOCR preferred, EasyOCR fallback)."""
    global _ocr_engine, _ocr_type, _paddle_api
    if _ocr_engine is not None:
        return _ocr_engine, _ocr_type

    engine, api_version = _init_paddleocr()
    if engine:
        _ocr_engine = engine
        _ocr_type = "paddle"
        _paddle_api = api_version
        return _ocr_engine, _ocr_type

    _ocr_engine = _init_easyocr()
    if _ocr_engine:
        _ocr_type = "easyocr"
        return _ocr_engine, _ocr_type

    return None, None


def _paddle_ocr_call(engine, img):
    """Call PaddleOCR with the correct API (v5 vs legacy).

    Returns list of (text, confidence) tuples.
    """
    global _paddle_api
    results_out = []

    if _paddle_api == "v5":
        try:
            gen = engine.predict(img)
            for result in gen:
                texts = result.get('rec_texts', [])
                scores = result.get('rec_scores', [])
                for text, score in zip(texts, scores):
                    results_out.append((text, score))
        except Exception:
            pass
    else:
        try:
            results = engine.ocr(img, cls=True)
            if results and results[0]:
                for line in results[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    results_out.append((text, conf))
        except Exception:
            # Maybe API changed — try predict as fallback
            try:
                _paddle_api = "v5"
                gen = engine.predict(img)
                for result in gen:
                    texts = result.get('rec_texts', [])
                    scores = result.get('rec_scores', [])
                    for text, score in zip(texts, scores):
                        results_out.append((text, score))
            except Exception:
                pass

    return results_out


def read_jersey_number(crop: np.ndarray) -> int | None:
    """
    Attempt to read a jersey number from a player crop.

    Returns the jersey number (1-99) or None if unreadable.
    """
    engine, ocr_type = get_ocr_engine()
    if engine is None:
        return None

    # Preprocess: enhance contrast, resize for better OCR
    processed = _preprocess_crop(crop)

    if ocr_type == "paddle":
        for text, conf in _paddle_ocr_call(engine, processed):
            num = _extract_jersey_number(text)
            if num is not None:
                return num
    elif ocr_type == "easyocr":
        try:
            results = engine.readtext(processed)
            for (_, text, conf) in results:
                num = _extract_jersey_number(text)
                if num is not None:
                    return num
        except Exception:
            pass

    return None


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance a player crop for better OCR results."""
    h, w = crop.shape[:2]

    # Upscale small crops more aggressively
    target_h = 128
    if h < target_h:
        scale = target_h / h
        crop = cv2.resize(crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (adaptive contrast) instead of global equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Bilateral filter to reduce noise while preserving edges (number outlines)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # Sharpen with a milder kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def read_jersey_number_multi(crop: np.ndarray) -> list[int]:
    """
    Try multiple preprocessing strategies and return all number candidates.

    Returns list of detected numbers (may contain duplicates for voting).
    """
    candidates = []

    # Strategy 1: Standard preprocessing
    num = read_jersey_number(crop)
    if num is not None:
        candidates.append(num)

    # Strategy 2: Binary threshold (white numbers on dark jersey)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 128:
        scale = 128 / h
        gray = cv2.resize(gray, (int(w * scale), 128), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    num2 = _read_from_preprocessed(binary_bgr)
    if num2 is not None:
        candidates.append(num2)

    # Strategy 3: Inverted binary (dark numbers on light jersey)
    inv_binary = cv2.bitwise_not(binary)
    inv_bgr = cv2.cvtColor(inv_binary, cv2.COLOR_GRAY2BGR)
    num3 = _read_from_preprocessed(inv_bgr)
    if num3 is not None:
        candidates.append(num3)

    return candidates


def _read_from_preprocessed(crop_bgr: np.ndarray) -> int | None:
    """Read jersey number from an already-preprocessed image."""
    engine, ocr_type = get_ocr_engine()
    if engine is None:
        return None

    if ocr_type == "paddle":
        for text, conf in _paddle_ocr_call(engine, crop_bgr):
            num = _extract_jersey_number(text)
            if num is not None:
                return num
    elif ocr_type == "easyocr":
        try:
            results = engine.readtext(crop_bgr)
            for (_, text, conf) in results:
                num = _extract_jersey_number(text)
                if num is not None:
                    return num
        except Exception:
            pass
    return None


def _extract_jersey_number(text: str) -> int | None:
    """Extract a valid jersey number (1-99) from OCR text."""
    digits = re.findall(r"\d+", text)
    for d in digits:
        num = int(d)
        if 1 <= num <= 99:
            return num
    return None


def get_dominant_color(crop: np.ndarray, top_fraction: float = 0.6) -> tuple:
    """
    Get the dominant jersey color from a player crop.

    Focuses on the upper portion of the crop (torso/jersey area).
    Returns (H, S, V) in OpenCV scale.
    """
    h, w = crop.shape[:2]
    jersey_region = crop[: int(h * top_fraction), :]
    hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    median_color = np.median(pixels, axis=0)
    return tuple(median_color.astype(int))


def color_to_name(hsv: tuple) -> str:
    """Convert an HSV color to a rough color name."""
    h, s, v = hsv
    if s < 40:
        if v < 80:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"
    if h < 10 or h > 165:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 80:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 165:
        return "purple"
    return "unknown"
