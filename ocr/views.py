import pytesseract
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import os
import platform


if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'


class OCRAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        image_files = request.FILES.getlist('images')  # ahora espera un array de imágenes

        if not image_files:
            return Response({'error': 'No se proporcionó ninguna imagen'}, status=400)

        results = []

        try:
            for image_file in image_files:
                # Preprocesamiento
                image = Image.open(image_file).convert('L')
                image = image.filter(ImageFilter.MedianFilter())
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2)

                # OpenCV
                image_np = np.array(image)
                _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # OCR
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(thresh, lang='spa', config=custom_config)

                results.append({
                    "filename": image_file.name,
                    "text": text.strip()
                })

            return Response({'results': results})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
