# download_models.py
import easyocr

# Esto descargará los modelos a la carpeta por defecto (~/.EasyOCR)
reader = easyocr.Reader(['es', 'en'], gpu=False)
