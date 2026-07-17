import os
import re
import cv2
import pytesseract
import zxingcpp
import threading
import numpy as np

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

# 修改成你的安裝路徑
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\10303707\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

SUPPORTED = (
    ".heic",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


class BarcodeRenamerGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Barcode Rename Tool")
        self.root.geometry("950x650")

        self.folder_var = StringVar()

        top = Frame(root)
        top.pack(fill=X, padx=10, pady=10)

        Entry(
            top,
            textvariable=self.folder_var,
            width=90
        ).pack(side=LEFT, padx=5)

        Button(
            top,
            text="選擇資料夾",
            command=self.select_folder
        ).pack(side=LEFT)

        Button(
            top,
            text="開始處理",
            command=self.start_process
        ).pack(side=LEFT, padx=5)

        self.progress = ttk.Progressbar(
            root,
            orient="horizontal",
            mode="determinate",
            length=900
        )
        self.progress.pack(pady=5)

        self.status_label = Label(
            root,
            text="Ready"
        )
        self.status_label.pack()

        self.log_text = Text(
            root,
            height=35
        )

        self.log_text.pack(
            fill=BOTH,
            expand=True,
            padx=10,
            pady=10
        )

    def log(self, msg):

        self.log_text.insert(
            END,
            msg + "\n"
        )

        self.log_text.see(END)

        self.root.update_idletasks()

    def select_folder(self):

        folder = filedialog.askdirectory()

        if folder:
            self.folder_var.set(folder)

    def start_process(self):

        threading.Thread(
            target=self.process_images,
            daemon=True
        ).start()

    # --------------------------
    # Barcode Retry
    # --------------------------
    def read_barcode_retry(self, img):

        scales = [1.0, 1.5, 2.0]

        rotations = [
            ("0", img),
            ("90", cv2.rotate(
                img, cv2.ROTATE_90_CLOCKWISE
            )),
            ("270", cv2.rotate(
                img,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )),
            ("180", cv2.rotate(
                img, cv2.ROTATE_180
            )),
        ]

        all_texts = []

        for rot_name, rot_img in rotations:

            for scale in scales:

                try:

                    resized = cv2.resize(
                        rot_img,
                        None,
                        fx=scale,
                        fy=scale
                    )

                    results = zxingcpp.read_barcodes(
                        resized,
                        try_rotate=True,
                        try_downscale=True,
                        try_invert=True,
                    )

                    for r in results:

                        txt = r.text.strip()

                        self.log(
                            f"ZXing (rot={rot_name}"
                            f" scale={scale}): "
                            f"{r.format} -> {txt}"
                        )

                        all_texts.append(txt)

                        match = re.search(
                            r"\d{13}",
                            txt
                        )

                        if match:

                            self.log(
                                f"ZXing取得13碼: "
                                f"{match.group()}"
                            )

                            return txt

                except Exception as e:

                    self.log(
                        f"ZXing失敗 "
                        f"(rot={rot_name}"
                        f" scale={scale}): {e}"
                    )

        if all_texts:
            return all_texts[0]

        return None

    # --------------------------
    # OCR 備援
    # --------------------------
    def ocr_find_13digit(self, img):

        h, w = img.shape[:2]

        rot90 = cv2.rotate(
            img, cv2.ROTATE_90_CLOCKWISE
        )
        rot270 = cv2.rotate(
            img,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        h90, w90 = rot90.shape[:2]

        regions = [
            ("下半部",
             img[int(h * 0.4):h, :]),
            ("右側",
             img[:, int(w * 0.6):]),
            ("全圖", img),
            ("rot90_下半部",
             rot90[int(h90 * 0.4):h90, :]),
            ("rot90_全圖", rot90),
            ("rot270_下半部",
             rot270[int(h90 * 0.4):h90, :]),
            ("rot270_全圖", rot270),
        ]

        for region_name, roi in regions:

            result = self._ocr_region(
                roi, region_name
            )

            if result:
                return result

        return None

    def _ocr_region(self, roi, region_name):

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        preprocess_list = []

        # 1. Otsu
        blur1 = cv2.GaussianBlur(
            gray, (3, 3), 0
        )
        _, otsu = cv2.threshold(
            blur1, 0, 255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )
        preprocess_list.append(
            ("Otsu", otsu)
        )

        # 2. CLAHE + Otsu
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        cl = clahe.apply(gray)
        _, cl_otsu = cv2.threshold(
            cl, 0, 255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )
        preprocess_list.append(
            ("CLAHE", cl_otsu)
        )

        # 3. Adaptive threshold
        blur2 = cv2.GaussianBlur(
            gray, (5, 5), 0
        )
        adapt = cv2.adaptiveThreshold(
            blur2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 4
        )
        preprocess_list.append(
            ("Adaptive", adapt)
        )

        # 4. 放大2倍 + CLAHE
        big = cv2.resize(
            gray, None,
            fx=2.0, fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )
        cl_big = clahe.apply(big)
        _, cl_big_otsu = cv2.threshold(
            cl_big, 0, 255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )
        preprocess_list.append(
            ("2x_CLAHE", cl_big_otsu)
        )

        psm_modes = ["6", "11", "3"]

        # Pass 1: 不加白名單，找真實的13位數字序列
        for prep_name, processed in preprocess_list:

            for psm in psm_modes:

                try:

                    config = f"--psm {psm}"

                    text = pytesseract.image_to_string(
                        processed,
                        config=config
                    )

                    match = re.search(
                        r"\d{13}",
                        text
                    )

                    if match:

                        self.log(
                            f"OCR[{region_name}]"
                            f"[{prep_name}]"
                            f"[psm{psm}]: "
                            f"{text.strip()[:80]}"
                        )

                        self.log(
                            f"OCR取得13碼: "
                            f"{match.group()}"
                        )

                        return match.group()

                except Exception:
                    pass

        # Pass 2: 白名單限定數字（備援）
        for prep_name, processed in preprocess_list:

            for psm in psm_modes:

                try:

                    config = (
                        f"--psm {psm} "
                        f"-c tessedit_char_whitelist="
                        f"0123456789"
                    )

                    text = pytesseract.image_to_string(
                        processed,
                        config=config
                    )

                    digits = re.sub(
                        r"\D", "", text
                    )

                    self.log(
                        f"OCR[{region_name}]"
                        f"[{prep_name}]"
                        f"[psm{psm}][WL]: "
                        f"{digits[:60]}"
                    )

                    match = re.search(
                        r"\d{13}",
                        digits
                    )

                    if match:

                        self.log(
                            f"OCR取得13碼: "
                            f"{match.group()}"
                        )

                        return match.group()

                except Exception as e:

                    self.log(
                        f"OCR失敗[{prep_name}]"
                        f"[psm{psm}]: {e}"
                    )

        return None

    # --------------------------
    # 主流程
    # --------------------------
    def process_images(self):

        folder = self.folder_var.get()

        if not os.path.exists(folder):

            self.log("資料夾不存在")

            return

        files = [

            f for f in os.listdir(folder)

            if f.lower().endswith(SUPPORTED)

        ]

        total = len(files)

        self.progress["maximum"] = total

        success = 0
        failed = 0
        skipped = 0

        for idx, filename in enumerate(files, start=1):

            fullpath = os.path.join(
                folder,
                filename
            )

            self.log("")
            self.log("=" * 60)
            self.log(f"Processing: {filename}")

            # 已經是13碼檔名直接跳過
            basename = os.path.splitext(
                filename
            )[0]

            if re.fullmatch(
                    r"\d{13}(_\d+)?",
                    basename):

                self.log(
                    "已是13碼檔名，跳過"
                )

                skipped += 1

                self.progress["value"] = idx

                continue

            try:

                pil_img = Image.open(
                    fullpath
                )

                if pil_img.mode != "RGB":

                    pil_img = pil_img.convert(
                        "RGB"
                    )

                img = cv2.cvtColor(
                    np.array(pil_img),
                    cv2.COLOR_RGB2BGR
                )

            except Exception as e:

                self.log(
                    f"讀圖失敗: {e}"
                )

                failed += 1

                continue

            found_13 = None

            # ------------------
            # ZXing
            # ------------------

            barcode_text = self.read_barcode_retry(
                img
            )

            if barcode_text:

                self.log(
                    f"Barcode: {barcode_text}"
                )

                match = re.search(
                    r"\d{13}",
                    barcode_text
                )

                if match:

                    found_13 = match.group()

                    self.log(
                        f"Barcode取得13碼: {found_13}"
                    )

            # ------------------
            # OCR 備援
            # ------------------

            if found_13 is None:

                self.log(
                    "ZXing未取得13碼，啟動OCR..."
                )

                found_13 = self.ocr_find_13digit(
                    img
                )

            # ------------------
            # 最終驗證
            # ------------------

            if found_13 is None:

                self.log(
                    "找不到13碼，放棄修改"
                )

                failed += 1

                continue

            if not re.fullmatch(
                    r"\d{13}",
                    found_13):

                self.log(
                    f"結果非13碼: {found_13}"
                )

                failed += 1

                continue

            try:

                new_name = (
                    f"{found_13}.jpg"
                )

                new_path = os.path.join(
                    folder,
                    new_name
                )

                dup = 1

                while os.path.exists(
                        new_path):

                    new_name = (
                        f"{found_13}_{dup}.jpg"
                    )

                    new_path = os.path.join(
                        folder,
                        new_name
                    )

                    dup += 1

                pil_img.save(
                    new_path,
                    "JPEG",
                    quality=95
                )

                os.remove(fullpath)

                self.log(
                    f"Rename -> {new_name}"
                )

                success += 1

            except Exception as e:

                self.log(
                    f"存檔失敗: {e}"
                )

                failed += 1

            self.progress["value"] = idx

        self.log("")
        self.log("=" * 60)
        self.log("全部完成")
        self.log(f"成功: {success}")
        self.log(f"失敗: {failed}")
        self.log(f"跳過: {skipped}")

        self.status_label.config(
            text=(
                f"完成 | "
                f"成功:{success} "
                f"失敗:{failed} "
                f"跳過:{skipped}"
            )
        )


if __name__ == "__main__":

    root = Tk()

    app = BarcodeRenamerGUI(root)

    root.mainloop()