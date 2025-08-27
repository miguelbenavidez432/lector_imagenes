from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import easyocr
import numpy as np
from PIL import Image
import io

class OCRAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        image_file = request.data.get('image')
        if not image_file:
            return Response({'error': 'No se proporcionó ninguna imagen'}, status=400)

        try:
            # Leer imagen como bytes y convertirla en numpy array
            image = Image.open(image_file)
            image = image.convert('RGB')  # Asegura que esté en formato RGB
            image_np = np.array(image)

            # Usar EasyOCR para procesar
            reader = easyocr.Reader(['es', 'en'])
            result = reader.readtext(image_np)

            # Extraer solo el texto
            text_result = [item[1] for item in result]
            return Response({'text': text_result})
        
        except Exception as e:
            return Response({'error': str(e)}, status=500)
