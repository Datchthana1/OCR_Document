from pdf2image import convert_from_bytes
import easyocr
import numpy as np
from typing import List
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API EasyOCR", description="PDF OCR Service with Thai and English support")

# Initialize EasyOCR reader at startup (not per request)
logger.info("Initializing EasyOCR reader...")
reader = easyocr.Reader(
    ['th','en'], 
    gpu=False, 
    model_storage_directory='/models/ocr',
    download_enabled=True
    )
logger.info("EasyOCR reader initialized successfully")

class PageContent(BaseModel):
    page_number: int
    content: str

class OCRResponse(BaseModel):
    pages: List[PageContent]
    full_text: str

@app.post("/ocr/pdf", response_model=OCRResponse)
async def pdf_ocr(request: Request):
    try:
        logger.info("Receiving PDF request...")
        pdf_bytes = await request.body()
        logger.info(f"PDF size: {len(pdf_bytes)} bytes")

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="No data received")

        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="Invalid PDF format")

        logger.info("Converting PDF to images...")
        pages = convert_from_bytes(pdf_bytes, dpi=350)
        logger.info(f"Converted {len(pages)} pages")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    all_text = []
    pages_data = []

    for i, page in enumerate(pages):
        logger.info(f"Processing page {i+1}/{len(pages)}...")
        result = reader.readtext(np.array(page))
        page_text = " ".join([text for _, text, _ in result])
        all_text.append(page_text)
        pages_data.append(PageContent(page_number=i+1, content=page_text))
        logger.info(f"Page {i+1} completed")

    logger.info("OCR processing completed successfully")
    return OCRResponse(
        pages=pages_data,
        full_text="\n".join(all_text)
    )

# @app.post("/ocr/msoffice", response_model=OCRResponse)
# def msoffice_ocr(request: Request):


@app.get("/")
async def root():
    return {"message": "OCR API is running", "endpoints": ["/ocr/pdf"]}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": reader is not None,
        "supported_languages": ['th', 'en']
    }
