import cv2
import numpy as np
import pytesseract
from rest_framework.decorators import api_view
from rest_framework.response import Response

def preprocess_image(image):
    """ Mejora contraste y hace binarización adaptativa para textos claros/oscuros """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Morfología para resaltar letras delgadas o claras
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed

def extract_text_from_zone(image, x, y, w, h, config=""):
    """ Recorta la zona, preprocesa y aplica OCR """
    roi = image[y:y+h, x:x+w]
    roi = preprocess_image(roi)
    text = pytesseract.image_to_string(roi, config=config).strip()
    return text

@api_view(["POST"])
def process(request):
    results = []
    files = request.FILES.getlist("images")

    for file in files:
        np_img = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        height, width, _ = image.shape

        # Definimos las columnas (ajusta según tu imagen)
        col_name = (50, 200, 400, 40)   # x, y, w, h
        col_sit  = (460, 200, 60, 40)
        col_gol  = (540, 200, 60, 40)
        col_asi  = (620, 200, 60, 40)
        col_con  = (700, 200, 60, 40)
        col_cal  = (780, 200, 80, 40)

        players = []
        row_height = 45  # separación entre filas

        for i in range(15):  # 15 filas
            offset_y = 70  # desplazar 70 px hacia arriba

            name = extract_text_from_zone(image, col_name[0], col_name[1] + i*row_height - offset_y, col_name[2], col_name[3])
            sit  = extract_text_from_zone(image, col_sit[0],  col_sit[1]  + i*row_height - offset_y, col_sit[2],  col_sit[3])
            gol  = extract_text_from_zone(image, col_gol[0],  col_gol[1]  + i*row_height - offset_y, col_gol[2],  col_gol[3])
            asi  = extract_text_from_zone(image, col_asi[0],  col_asi[1]  + i*row_height - offset_y, col_asi[2],  col_asi[3])
            con  = extract_text_from_zone(image, col_con[0],  col_con[1]  + i*row_height - offset_y, col_con[2],  col_con[3])
            cal  = extract_text_from_zone(image, col_cal[0],  col_cal[1]  + i*row_height - offset_y, col_cal[2],  col_cal[3])

            # Normalizar gol: contar símbolos en vez de texto
            gol_count = len([c for c in gol if not c.isalnum() and not c.isspace()])

            # Normalizar cal: quedarnos solo con números (última columna)
            cal_num = "".join([c for c in cal if c.isdigit()])

            players.append({
                "name": name,
                "sit": sit,
                "gol": gol_count,
                "asi": asi,
                "con": con,
                "cal": cal_num
            })

        results.append({
            "filename": file.name,
            "players": players
        })

    return Response({"results": results})
