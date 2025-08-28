# ocr/views.py
import io
import os
import platform
import re
from typing import List, Dict, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

# --- Tesseract en Windows local (ignorado en Linux/Render) ---
if platform.system() == "Windows":
    # Ajustá estas rutas si tu instalación es distinta
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# ---------- Utilidades de imagen ----------

OCR_CONFIG = "--oem 3 --psm 6 -l spa+eng"

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB/LA/L → OpenCV BGR/GRAYSCALE."""
    if pil_img.mode not in ("L", "RGB"):
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    if pil_img.mode == "RGB":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr  # L → gris

def _deskew(gray: np.ndarray) -> np.ndarray:
    """Intento rápido de enderezar (deskew) si hay inclinación."""
    try:
        # Umbral para detectar bordes y líneas principales
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thr = 255 - thr  # invertimos para que texto sea blanco
        coords = np.column_stack(np.where(thr > 0))
        if coords.size == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        # Si la inclinación es pequeña, no tocamos
        if abs(angle) < 0.8:
            return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Preprocesamiento OCR optimizado con OpenCV:
    - Escala de grises
    - Suavizado leve (reduce ruido sin borrar letras)
    - CLAHE (mejor contraste local)
    - Binarización adaptativa
    - Inversión (texto oscuro sobre fondo claro)
    - Dilatación suave (engrosar letras delgadas)
    """
    cv_img = _pil_to_cv2(pil_img)

    # Paso 1: escala de grises
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

    # Paso 2: Suavizado sin perder bordes (reduce ruido sin borrar texto)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Paso 3: Contraste local (mejora letras sobre fondo sucio)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Paso 4: Binarización adaptativa
    bw = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    # Paso 5: Invertimos (texto blanco sobre negro)
    bw = 255 - bw

    # Paso 6: Dilatación leve horizontal (letras finas o partidas)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    bw = cv2.dilate(bw, kernel, iterations=1)

    return bw


# ---------- OCR + parsing ----------

SCALE = 1.0  # en lugar de 2.0
ORIGINAL_Y1, ORIGINAL_Y2 = 140, 615

# Coordenadas globales para columnas
y1, y2 = int(ORIGINAL_Y1 * SCALE), int(ORIGINAL_Y2 * SCALE)  # Coordenadas Y comunes a todas las columnas

columns = {
    "sit":    (int(105 * SCALE), int(130 * SCALE)),
    "nombre": (int(131 * SCALE), int(650 * SCALE)),
    "gol":   (int(1214 * SCALE), int(1239 * SCALE)),
    "asi":   (int(1240 * SCALE), int(1263 * SCALE)),
    "con":   (int(1264 * SCALE), int(1288 * SCALE)),
    "cal":   (int(1289 * SCALE), int(1338 * SCALE)),
}


NAME_REGEX = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b"
)
RATING_REGEX = re.compile(r"\b([0-9]\.[0-9])\b")  # 6.3, 7.4, 8.0, etc.
BAD_TOKENS = {"DATOS", "ROMA", "CHELSEA", "Estadísticas", "NOMBRE", "MIN", "SIT"}

def ocr_with_data(img: np.ndarray) -> Dict[str, Any]:
    """
    Ejecuta Tesseract y devuelve:
      - raw_text
      - lines: lista de dicts con 'text', 'conf', 'y', 'x'
    """
    config = "--oem 3 --psm 6 -l spa+eng"
    # Texto completo
    raw_text = pytesseract.image_to_string(img, config=config)

    # Datos por palabra (para reconstruir líneas)
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    lines: Dict[tuple, Dict[str, Any]] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        entry = lines.setdefault(key, {"words": [], "conf": [], "xs": [], "ys": []})
        entry["words"].append(text)
        entry["conf"].append(conf)
        entry["xs"].append(data["left"][i])
        entry["ys"].append(data["top"][i])

    line_list = []
    for (_, _, _), info in lines.items():
        line_text = " ".join(info["words"])
        avg_conf = float(np.mean([c for c in info["conf"] if c >= 0])) if info["conf"] else -1.0
        y = int(np.median(info["ys"])) if info["ys"] else 0
        x = int(min(info["xs"])) if info["xs"] else 0
        line_list.append({"text": line_text, "conf": avg_conf, "y": y, "x": x})

    # ordenamos top→down
    line_list.sort(key=lambda d: d["y"])
    return {"raw_text": raw_text, "lines": line_list}

def extract_column_cells(image: np.ndarray, x1: int, x2: int, y1: int, y2: int, num_rows: int) -> List[np.ndarray]:
    """
    Divide una zona de la imagen en celdas horizontales correspondientes a filas de una columna.
    """

    height, width = image.shape[:2]

    # Validación: asegurarse de que las coordenadas estén dentro de la imagen
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop coordinates: ({x1}, {x2}) x ({y1}, {y2}) for image size {width}x{height}")

    height = y2 - y1
    row_height = height // num_rows
    cells = []

    for i in range(num_rows):
        y_start = y1 + i * row_height
        y_end = y_start + row_height

        # Asegurar que no se pase del borde inferior
        if y_end > height + y1:
            break

        cell = image[y_start:y_end, x1:x2]
        cells.append(cell)

    return cells


def extract_text_from_cells(cells: List[np.ndarray]) -> List[str]:
    """
    Aplica OCR a cada celda individualmente y extrae el texto plano.

    Returns:
        Lista de strings, uno por celda.
    """
    results = []
    for cell in cells:
        text = pytesseract.image_to_string(cell, config=OCR_CONFIG).strip()
        results.append(text)
    return results

def extract_stats_from_image(image: np.ndarray, num_rows: int = 20) -> List[Dict[str, str]]:
    """
    Extrae datos de tabla usando detección de líneas con OpenCV.
    """
    # Paso 1: inverso y binarizar si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invertimos si es texto blanco sobre negro
    mean_val = np.mean(gray)
    if mean_val < 127:
        gray = 255 - gray

    # Binarizar
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Paso 2: detectar líneas horizontales y verticales
    horizontal = bw.copy()
    vertical = bw.copy()

    # Kernel horizontal (líneas de filas)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.erode(horizontal, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=1)

    # Kernel vertical (líneas de columnas)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical = cv2.erode(vertical, v_kernel, iterations=1)
    vertical = cv2.dilate(vertical, v_kernel, iterations=1)

    # Paso 3: encontrar intersecciones (esquinas de celdas)
    mask = cv2.bitwise_and(horizontal, vertical)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Paso 4: detectar bounding boxes (celdas)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # orden: y, luego x

    # Agrupar por filas
    rows = []
    current_row = []
    last_y = -1
    tolerance = 10

    for box in boxes:
        x, y, w, h = box
        if last_y == -1 or abs(y - last_y) <= tolerance:
            current_row.append(box)
            last_y = y
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            last_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # Paso 5: extraer texto por celda
    player_dicts = []
    for row in rows:
        player = {}
        for idx, (x, y, w, h) in enumerate(row):
            cell = gray[y:y + h, x:x + w]
            cell = cv2.copyMakeBorder(cell, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            text = pytesseract.image_to_string(cell, config=OCR_CONFIG).strip()
            player[f"col_{idx}"] = text
        player_dicts.append(player)

    return player_dicts


def ocr_from_roi(proc: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> List[Dict[str, Any]]:
    """
    Realiza OCR en una región específica de la imagen.
    Retorna líneas de texto con coordenadas relativas al recorte.
    """
    roi = proc[y1:y2, x1:x2]
    roi = cv2.resize(roi, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    data = pytesseract.image_to_data(roi, config="--oem 3 --psm 6 -l spa+eng", output_type=pytesseract.Output.DICT)

    lines = {}
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        entry = lines.setdefault(key, {"words": [], "ys": [], "xs": []})
        entry["words"].append(txt)
        entry["ys"].append(data["top"][i])
        entry["xs"].append(data["left"][i])

    result_lines = []
    for info in lines.values():
        y = int(np.median(info["ys"]))
        x = int(min(info["xs"]))
        result_lines.append({"text": " ".join(info["words"]), "y": y + y1, "x": x + x1})

    result_lines.sort(key=lambda d: d["y"])
    return result_lines


def parse_players(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Heurística para extraer jugadores:
      - Buscar líneas con nombre propio (2–4 palabras capitalizadas).
      - Ignorar líneas que son encabezados.
      - Tomar la última calificación X.Y en la línea como 'rating' (si existe).
    """
    players = []
    seen = set()

    for ln in lines:
        text = ln["text"]
        # filtramos encabezados ruidosos
        if any(h in text for h in BAD_TOKENS):
            continue

        name_match = NAME_REGEX.search(text)
        if not name_match:
            continue

        name = name_match.group(1).strip()
        # Evitar falsos positivos muy cortos (p.ej. "Junior")
        if len(name.split()) < 2:
            continue

        # Calificación: último número tipo 7.4 en la línea
        rating = None
        ratings = RATING_REGEX.findall(text)
        if ratings:
            try:
                rating = float(ratings[-1])
                if rating < 5.0 or rating > 10.0:
                    # descartar valores absurdos
                    rating = None
            except Exception:
                rating = None

        key = name.lower()
        if key in seen:
            # si ya existe, actualizamos rating si estaba vacío
            for p in players:
                if p["name"].lower() == key and p.get("rating") is None and rating is not None:
                    p["rating"] = rating
            continue

        players.append({"name": name, "rating": rating})
        seen.add(key)

    return players

# ---------- Vista principal ----------

class OCRAPIView(APIView):
    """
    POST /api/ocr/
      Form-data:
        - image (una sola)  o  images[] (múltiples)
      Respuesta:
      {
        "results": [
          {
            "filename": "...",
            "raw_text": "...",
            "players": [{"name":"...", "rating": 7.4}, ...]
          },
          ...
        ],
        "consolidated_players": [{"name":"...", "rating": 7.4}, ...]  # union por nombre
      }
    """
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        files = []
        if "image" in request.FILES:
            files.append(request.FILES["image"])
        files.extend(request.FILES.getlist("images"))

        if not files:
            return Response({"error": "No se proporcionó ninguna imagen"}, status=400)

        all_results = []

        try:
            for f in files:
                pil = Image.open(f.stream if hasattr(f, "stream") else f).convert("RGB")
                proc = preprocess(pil)

                if proc is None or proc.shape[0] < y2 or proc.shape[1] < max(x2 for (_, x2) in columns.values()):
                    return Response({"error": "La imagen es demasiado pequeña para los rangos definidos"}, status=400)


                debug = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                for key, (x1, x2) in columns.items():
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite("debug_zonas_original_scale.jpg", debug)


                # Guardar para debug (opcional)
                cv2.imwrite("debug_proc.jpg", proc)

                # Usar función correcta que extrae columnas por coordenadas
                players_stats = extract_stats_from_image(proc, num_rows=20)

                all_results.append({
                    "filename": getattr(f, "name", "image"),
                    "players": players_stats
                })

            return Response({"results": all_results})

        except Exception as e:
            return Response({"error": str(e)}, status=500)

