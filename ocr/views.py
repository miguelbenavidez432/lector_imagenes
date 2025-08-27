from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import easyocr
import numpy as np
from PIL import Image
import io
import os

MODEL_DIR = 'models'

EXPECTED_MODELS = [
    'craft_mlt_25k.pth',
    'latin_g2.pth'
]

missing = [f for f in EXPECTED_MODELS if not os.path.isfile(os.path.join(MODEL_DIR, f))]
if missing:
    raise FileNotFoundError(f"Faltan los siguientes modelos en '{MODEL_DIR}': {missing}")

reader = easyocr.Reader(
    ['es', 'en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=False
)

class OCRAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No se proporcionó ninguna imagen'}, status=400)

        try:
            image = Image.open(image_file)
            image = image.convert('RGB')
            image_np = np.array(image)

            result = reader.readtext(image_np)

            return Response({'text': [item[1] for item in result]})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
