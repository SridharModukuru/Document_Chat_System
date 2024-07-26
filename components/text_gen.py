import os
from PyPDF2 import PdfReader


def get_pdf_text(UPLOAD_FOLDER):
    text = ''
    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER,file_name)
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text
