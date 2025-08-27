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
    Preprocesamiento para mejorar contraste sobre fondo oscuro:
    - Convertir a gris
    - CLAHE (aumentar contraste local)
    - Binarizado adaptativo
    - Invertido (texto negro → blanco)
    - Dilatación suave para engrosar trazos finos
    """
    cv_img = _pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

    # Upscale x1.5 para dar más info al OCR
    h, w = gray.shape[:2]
    gray = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)

    # Enderezar si es necesario
    gray = _deskew(gray)

    # Desruido leve + CLAHE
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarizado adaptativo + invertir
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    bw = 255 - bw

    # Dilatación horizontal suave ayuda a que números 7.4 no se corten
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    bw = cv2.dilate(bw, kernel, iterations=1)

    return bw

# ---------- OCR + parsing ----------

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
        # soportamos image o images[]
        files = []
        if "image" in request.FILES:
            files.append(request.FILES["image"])
        files.extend(request.FILES.getlist("images"))

        if not files:
            return Response({"error": "No se proporcionó ninguna imagen"}, status=400)

        all_results = []
        by_name_best: Dict[str, Dict[str, Any]] = {}

        try:
            for f in files:
                pil = Image.open(f.stream if hasattr(f, "stream") else f).convert("RGB")
                proc = preprocess(pil)

                ocr_data = ocr_with_data(proc)
                players = parse_players(ocr_data["lines"])

                # construir respuesta por imagen
                all_results.append({
                    "filename": getattr(f, "name", "image"),
                    "raw_text": ocr_data["raw_text"],
                    "players": players,
                })

                # consolidado por nombre: mantenemos el mejor (con rating no nulo)
                for p in players:
                    key = p["name"].lower()
                    best = by_name_best.get(key)
                    if best is None or (best.get("rating") is None and p.get("rating") is not None):
                        by_name_best[key] = {"name": p["name"], "rating": p.get("rating")}

            consolidated = list(by_name_best.values())
            return Response({"results": all_results, "consolidated_players": consolidated})

        except Exception as e:
            return Response({"error": str(e)}, status=500)
