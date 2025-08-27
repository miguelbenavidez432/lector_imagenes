import pytesseract
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import os

# Configurar ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

class OCRAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No se proporcionó ninguna imagen'}, status=400)

        try:
            # Abrir y preprocesar la imagen
            image = Image.open(image_file).convert('L')  # Escala de grises
            image = image.filter(ImageFilter.MedianFilter())
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)

            # Convertir PIL a array OpenCV
            image_np = np.array(image)
            _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Aplicar OCR con idioma español
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, lang='spa', config=custom_config)

            return Response({'text': text})
        except Exception as e:
            return Response({'error': str(e)}, status=500)
