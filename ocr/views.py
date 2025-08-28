import cv2
import numpy as np
import pytesseract
from rest_framework.decorators import api_view
from rest_framework.response import Response

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed

import os

def extract_text_from_zone(image, x, y, w, h, config="", label="", row=0, debug_dir="debug_rois"):
    # asegurar que no se pase de los límites
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x1 >= x2 or y1 >= y2:
        return ""  # ROI vacío

    roi = image[y1:y2, x1:x2]
    processed = preprocess_image(roi)

    # --- DEBUG: guardar ROI original y procesado ---
    os.makedirs(debug_dir, exist_ok=True)
    #cv2.imwrite(os.path.join(debug_dir, f"row{row}_{label}_orig.png"), roi)
    #cv2.imwrite(os.path.join(debug_dir, f"row{row}_{label}_proc.png"), processed)

    # OCR
    text = pytesseract.image_to_string(processed, config=config).strip()
    return text


@api_view(["POST"])
def process(request):
    results = []
    files = request.FILES.getlist("images")

    for file in files:
        np_img = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        height, width, _ = image.shape

        # -------------------------
        # Definir filas y columnas
        # -------------------------
        start_y = 225     # comienzo tabla
        row_height = 24   # altura por fila
        num_rows = 23     # jugadores visibles

        # columnas relativas a cada fila
        cols = {
            "sit":  (104, 32),
            "name": (150, 360),   # x, ancho
            "gol":  (1214, 26),
            "asi":  (1240, 23),
            #"con":  (1264, 26),
            "cal":  (1291, 39),
        }

        players = []

        for i in range(num_rows):
            y = max(0, start_y + i * row_height - 70)  # aplicar offset

            # cortar cada campo por fila
            for i in range(num_rows):
                y = max(0, start_y + i * row_height - 70)

                name = extract_text_from_zone(image, cols["name"][0], y, cols["name"][1], row_height, label="name", row=i)
                sit  = extract_text_from_zone(image, cols["sit"][0],  y, cols["sit"][1],  row_height, label="sit", row=i)
                gol  = extract_text_from_zone(image, cols["gol"][0],  y, cols["gol"][1],  row_height, label="gol", row=i)
                asi  = extract_text_from_zone(image, cols["asi"][0],  y, cols["asi"][1],  row_height, label="asi", row=i)
                #con  = extract_text_from_zone(image, cols["con"][0],  y, cols["con"][1],  row_height, label="con", row=i)
                cal  = extract_text_from_zone(image, cols["cal"][0],  y, cols["cal"][1],  row_height, label="cal", row=i)


            # Normalización
            gol_count = len([c for c in gol if not c.isalnum() and not c.isspace()])
            asi_count = len([c for c in asi if not c.isalnum() and not c.isspace()])
            #con_count = len([c for c in con if not c.isalnum() and not c.isspace()])
            cal_num   = "".join([c for c in cal if c.isdigit() or c == "."])

            players.append({
                "name": name,
                "sit": sit,
                "gol": gol_count,
                "asi": asi_count,
                #"con": con_count,
                "cal": cal_num
            })

            # Debug visual → dibujar rectángulos en la imagen
            cv2.rectangle(image, (cols["name"][0], y), (cols["name"][0]+cols["name"][1], y+row_height), (0,255,0), 1)
            cv2.rectangle(image, (cols["sit"][0], y),  (cols["sit"][0]+cols["sit"][1], y+row_height), (0,255,0), 1)
            cv2.rectangle(image, (cols["gol"][0], y),  (cols["gol"][0]+cols["gol"][1], y+row_height), (0,255,0), 1)
            cv2.rectangle(image, (cols["asi"][0], y),  (cols["asi"][0]+cols["asi"][1], y+row_height), (0,255,0), 1)
            #cv2.rectangle(image, (cols["con"][0], y),  (cols["con"][0]+cols["con"][1], y+row_height), (0,255,0), 1)
            cv2.rectangle(image, (cols["cal"][0], y),  (cols["cal"][0]+cols["cal"][1], y+row_height), (0,255,0), 1)

        results.append({
            "filename": file.name,
            "players": players
        })

        # guardar imagen debug
        cv2.imwrite("debug_filas.png", image)

    return Response({"results": results})
