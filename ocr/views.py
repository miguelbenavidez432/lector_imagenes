# ocr/views.py
import io
import os
import platform
import re
import pytesseract
import cv2
import numpy as np
import statistics
from typing import List, Dict, Any
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

REF_WIDTH = 1366
REF_HEIGHT = 768

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
    Preprocesamiento OCR mejorado:
    - Escala de grises
    - Aumento de resolución
    - Filtro gaussiano suave
    - Mejora de contraste (CLAHE)
    - Binarización Otsu (no adaptativa tan agresiva)
    - Sin invertir ni dilatar (dejamos letras limpias)
    """
    cv_img = _pil_to_cv2(pil_img)

    # Escala de grises
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

    # Reescalar para ganar detalle
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Suavizado leve
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarización Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bw



# ---------- OCR + parsing ----------

SCALE = 1.0  # en lugar de 2.0
ORIGINAL_Y1, ORIGINAL_Y2 = 200, 730

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

    # Paso 1: Binarizamos sin invertir aún
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertimos si es texto blanco sobre negro
    if np.mean(binary) < 127:
        binary = 255 - binary

    # Binarizar
    #_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Paso 2: detectar líneas horizontales y verticales
    horizontal = binary.copy()
    vertical = binary.copy()

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # ancho de líneas horizontales
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))  # alto de líneas verticales

    horizontal = cv2.erode(horizontal, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=1)

    vertical = cv2.erode(vertical, v_kernel, iterations=1)
    vertical = cv2.dilate(vertical, v_kernel, iterations=1)
   
    # Paso 3: encontrar intersecciones (esquinas de celdas)
     # Paso 3: Intersecciones (esquinas de celdas)
    table_mask = cv2.bitwise_and(horizontal, vertical)

    # Para visualizar la detección (debug opcional)
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Agrupamos por filas usando tolerancia vertical
    rows = []
    current_row = []
    last_y = -1
    tolerance = 10

    for box in boxes:
        x, y, w, h = box
        if w < 20 or h < 15:
            continue  # ignorar cajas demasiado pequeñas (ruido)

        if last_y == -1 or abs(y - last_y) <= tolerance:
            current_row.append(box)
            last_y = y
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            last_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # Paso 4: extraer texto por celda
    results = []
    for row in rows:
        player = {}
        for idx, (x, y, w, h) in enumerate(row):
            cell = gray[y:y + h, x:x + w]
            cell = cv2.copyMakeBorder(cell, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

            text = pytesseract.image_to_string(cell, config=OCR_CONFIG).strip()

            player[f"col_{idx}"] = text

            # Para visualizar (opcional)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if player:
            results.append(player)

    # Guardar imagen de debug
    cv2.imwrite("debug_detected_cells.jpg", debug_img)

    return results


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

def scale_coords_for_image(img_w: int, img_h: int, columns: dict):
    """Devuelve un nuevo dict de columnas y y1/y2 escaladas a la imagen actual."""
    sx = img_w / REF_WIDTH
    sy = img_h / REF_HEIGHT
    scaled = {}
    for k, (x1, x2) in columns.items():
        scaled[k] = (int(round(x1 * sx)), int(round(x2 * sx)))
    return scaled, int(round(ORIGINAL_Y1 * sy)), int(round(ORIGINAL_Y2 * sy))

def ensure_ocr_orientation(img: np.ndarray) -> np.ndarray:
    """
    Asegura que la imagen para Tesseract tenga texto oscuro sobre fondo claro.
    Si recibimos texto claro sobre oscuro, lo invertimos.
    """
    # img se espera GRAY (uint8)
    mean_val = np.mean(img)
    # Si la media es baja (imagen mayormente negra), invertimos para que tenga fondo claro.
    if mean_val < 127:
        return 255 - img
    return img

def extract_name_lines(ocr_img: np.ndarray, x1: int, x2: int, y1: int, y2: int):
    """
    Extrae líneas (texto y coordenada Y) de la columna 'nombre' usando image_to_data.
    Retorna lista ordenada por Y: [{"text": "...", "y": ..., "x": ...}, ...]
    """
    roi = ocr_img[y1:y2, x1:x2]
    # Upscale para mejorar OCR en UI pequeñas
    roi_resized = cv2.resize(roi, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(roi_resized, config="--oem 3 --psm 6 -l spa+eng", output_type=pytesseract.Output.DICT)

    lines = {}
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt:
            continue
        # coordenadas ya en ROI_resized — convertimos a ROI unit scale
        top = int(data["top"][i] / 2.5)  # revertir el upscale para mantener referencia en coords originales del ROI
        left = int(data["left"][i] / 2.5)
        key = data["line_num"][i]  # grouping por línea en el ROI_resized
        entry = lines.setdefault(key, {"words": [], "tops": [], "lefts": []})
        entry["words"].append(txt)
        entry["tops"].append(top)
        entry["lefts"].append(left)

    out = []
    for info in lines.values():
        line_text = " ".join(info["words"])
        y_rel = int(np.median(info["tops"]))
        x_rel = int(min(info["lefts"])) if info["lefts"] else 0
        out.append({"text": line_text, "y": y_rel + y1, "x": x_rel + x1})

    out.sort(key=lambda d: d["y"])
    return out

def ocr_cell_single_line(image_gray: np.ndarray, x1: int, x2: int, y_center: int, row_h: int):
    y_top = max(0, int(y_center - row_h // 2 - 3))
    y_bot = min(image_gray.shape[0], int(y_center + row_h // 2 + 3))
    x1 = max(0, x1)
    x2 = min(image_gray.shape[1], x2)

    # recorte de la celda
    cell = image_gray[y_top:y_bot, x1:x2]

    if cell.size == 0:
        return ""

    # upscale para mejorar OCR
    cell = cv2.resize(cell, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # pequeño preprocesado
    _, cell_bw = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(cell_bw) < 127:  # invertir si quedó blanco sobre negro
        cell_bw = 255 - cell_bw

    txt = pytesseract.image_to_string(
        cell_bw,
        config="--oem 3 --psm 7 -l spa+eng"
    )
    return txt.strip()


def extract_table_by_name_reference(image_gray: np.ndarray, columns_scaled: dict, y1: int, y2: int):
    """
    Strategy:
      - extraer líneas en columna 'nombre'
      - estimar altura de fila por mediana de diffs de Y
      - para cada fila, recortar otras columnas y OCR (psm 7)
    """
    name_x1, name_x2 = columns_scaled["nombre"]
    names = extract_name_lines(image_gray, name_x1, name_x2, y1, y2)

    if not names:
        return []  # fallback más abajo

    # Calcular row_h (estimación de altura de fila)
    ys = [n["y"] for n in names]
    if len(ys) >= 2:
        diffs = [j - i for i, j in zip(ys[:-1], ys[1:]) if j - i > 3]
        row_h = int(statistics.median(diffs)) if diffs else 30
    else:
        row_h = 30

    results = []
    for nm in names:
        row_y = nm["y"]
        row = {"name": nm["text"]}

        for key in columns_scaled:
            if key == "nombre":
                continue
            x1, x2 = columns_scaled[key]
            val = ocr_cell_single_line(image_gray, x1, x2, row_y, row_h)

            # --- DEBUG OPCIONAL ---
            cell_dbg = image_gray[max(0, row_y - row_h//2): row_y + row_h//2, x1:x2]
            os.makedirs("debug_cells", exist_ok=True)
            cv2.imwrite(f"debug_cells/{key}_{nm['text']}.jpg", cell_dbg)
            # ----------------------

            row[key] = val
        results.append(row)

    return results

# ---------- Vista principal ----------

class OCRAPIView(APIView):
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
                proc = preprocess(pil)  # tu preprocess retorna BINARIA (en mi versión original era texto blanco sobre negro)

                # convertir a gris si es binaria a color
                if len(proc.shape) == 3:
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                else:
                    gray = proc.copy()

                img_h, img_w = gray.shape[:2]
                columns_scaled, sy1, sy2 = scale_coords_for_image(img_w, img_h, columns)

                # preparamos imagen para OCR: texto oscuro sobre fondo claro (Tesseract prefiere así)
                ocr_ready = ensure_ocr_orientation(gray)

                # guarda debug de las zonas escaladas
                debug = cv2.cvtColor(ocr_ready, cv2.COLOR_GRAY2BGR)
                for key, (x1, x2) in columns_scaled.items():
                    cv2.rectangle(debug, (x1, sy1), (x2, sy2), (0, 255, 0), 2)
                cv2.imwrite("debug_zonas_scaled.jpg", debug)
                cv2.imwrite("debug_proc.jpg", ocr_ready)

                # EXTRACCIÓN usando columna 'nombre' como referencia
                players_stats = extract_table_by_name_reference(ocr_ready, columns_scaled, sy1, sy2)

                # Si no encontramos nombres (fallback): intentar OCR global agrupado por Y
                if not players_stats:
                    # fallback simple: OCR full area y agrupar por Y (más costoso)
                    full_roi = ocr_ready[sy1:sy2, :]
                    data = pytesseract.image_to_data(full_roi, config=OCR_CONFIG, output_type=pytesseract.Output.DICT)
                    # ... podrías implementar grouping similar; por ahora devolvemos raw_text
                    full_text = pytesseract.image_to_string(full_roi, config=OCR_CONFIG)
                    all_results.append({
                        "filename": getattr(f, "name", "image"),
                        "raw_text": full_text,
                        "players": []
                    })
                else:
                    all_results.append({
                        "filename": getattr(f, "name", "image"),
                        "players": players_stats
                    })

            return Response({"results": all_results})

        except Exception as e:
            return Response({"error": str(e)}, status=500)