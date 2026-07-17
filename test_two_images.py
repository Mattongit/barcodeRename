"""
Quick test: check if the two HEIC images can be recognized.
Does NOT rename files, only prints detection results.
"""
import os
import re
import cv2
import pytesseract
import zxingcpp
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\10303707\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

TEST_FILES = [
    r"c:\Python313\Scripts\images\IMG_6547.HEIC",
    r"c:\Python313\Scripts\images\IMG_6536.HEIC",
]


def read_barcode_retry(img):

    scales = [1.0, 1.5, 2.0]

    rotations = [
        ("0", img),
        ("90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("180", cv2.rotate(img, cv2.ROTATE_180)),
    ]

    all_texts = []

    for rot_name, rot_img in rotations:
        for scale in scales:
            try:
                resized = cv2.resize(
                    rot_img, None, fx=scale, fy=scale
                )
                results = zxingcpp.read_barcodes(
                    resized,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
                for r in results:
                    txt = r.text.strip()
                    print(f"  ZXing (rot={rot_name} scale={scale}): {r.format} -> {txt}")
                    all_texts.append(txt)
                    match = re.search(r"\d{13}", txt)
                    if match:
                        print(f"  >>> ZXing取得13碼: {match.group()}")
                        return txt
            except Exception as e:
                print(f"  ZXing失敗 (rot={rot_name} scale={scale}): {e}")

    if all_texts:
        return all_texts[0]
    return None


def ocr_find_13digit(img):

    h, w = img.shape[:2]
    rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h90, w90 = rot90.shape[:2]

    regions = [
        ("下半部", img[int(h * 0.4):h, :]),
        ("右側", img[:, int(w * 0.6):]),
        ("全圖", img),
        ("rot90_下半部", rot90[int(h90 * 0.4):h90, :]),
        ("rot90_全圖", rot90),
        ("rot270_下半部", rot270[int(h90 * 0.4):h90, :]),
        ("rot270_全圖", rot270),
    ]

    for region_name, roi in regions:
        result = _ocr_region(roi, region_name)
        if result:
            return result
    return None


def _ocr_region(roi, region_name):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cl = clahe.apply(gray)
    _, cl_otsu = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
    adapt = cv2.adaptiveThreshold(
        blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )

    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cl_big = clahe.apply(big)
    _, cl_big_otsu = cv2.threshold(cl_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    preprocess_list = [
        ("Otsu", otsu),
        ("CLAHE", cl_otsu),
        ("Adaptive", adapt),
        ("2x_CLAHE", cl_big_otsu),
    ]

    psm_modes = ["6", "11", "3"]

    # Pass 1: no whitelist
    for prep_name, processed in preprocess_list:
        for psm in psm_modes:
            try:
                config = f"--psm {psm}"
                text = pytesseract.image_to_string(processed, config=config)
                match = re.search(r"\d{13}", text)
                if match:
                    print(f"  OCR[{region_name}][{prep_name}][psm{psm}]: {text.strip()[:80]}")
                    print(f"  >>> OCR取得13碼: {match.group()}")
                    return match.group()
            except Exception:
                pass

    # Pass 2: whitelist
    for prep_name, processed in preprocess_list:
        for psm in psm_modes:
            try:
                config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(processed, config=config)
                digits = re.sub(r"\D", "", text)
                match = re.search(r"\d{13}", digits)
                if match:
                    print(f"  OCR[{region_name}][{prep_name}][psm{psm}][WL]: {digits[:60]}")
                    print(f"  >>> OCR取得13碼(WL): {match.group()}")
                    return match.group()
            except Exception:
                pass

    return None


if __name__ == "__main__":
    for filepath in TEST_FILES:
        print("=" * 60)
        print(f"Testing: {os.path.basename(filepath)}")
        print("=" * 60)

        pil_img = Image.open(filepath)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        found_13 = None

        barcode_text = read_barcode_retry(img)
        if barcode_text:
            match = re.search(r"\d{13}", barcode_text)
            if match:
                found_13 = match.group()

        if found_13 is None:
            print("  ZXing未取得13碼，啟動OCR...")
            found_13 = ocr_find_13digit(img)

        print()
        if found_13 and re.fullmatch(r"\d{13}", found_13):
            print(f"  *** RESULT: {found_13} ***")
        else:
            print(f"  *** FAILED: cannot find 13 digits ***")
        print()
