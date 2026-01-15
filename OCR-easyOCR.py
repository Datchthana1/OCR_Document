from pdf2image import convert_from_bytes
import easyocr
import numpy as np
import cv2
import torch
from typing import List
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging
from langchain_core.documents import Document

torch.set_num_threads(4)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR API EasyOCR",
    description="Optimized PDF OCR Service (Thai + English)"
)

reader = easyocr.Reader(
    ['th', 'en'],
    gpu=False
)

class PageContent(BaseModel):
    page_number: int
    content: str

class OCRResponse(BaseModel):
    text: str
    pages: list[dict]

@app.post("/ocr/pdf", response_model=OCRResponse)
async def pdf_ocr(request: Request):
    pdf_bytes = await request.body()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="No data received")

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF format")

    pages = convert_from_bytes(pdf_bytes, dpi=180)

    pages_data: List[PageContent] = []
    all_text: list[str] = []

    for i, page in enumerate(pages):
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape
        if w > 1600:
            scale = 1600 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        result = reader.readtext(
            gray,
            detail=0,
            paragraph=False
        )

        page_text = " ".join(result).strip()

        if page_text:
            pages_data.append(
                PageContent(
                    page_number=i + 1,
                    content=page_text
                )
            )
            all_text.append(page_text)

    full_text = "\n".join(all_text).strip()

    if not full_text:
        raise HTTPException(
            status_code=422,
            detail="OCR failed: no readable text found in PDF"
        )

    lc_document = Document(
        page_content=full_text,
        metadata={
            "source": "pdf_ocr",
            "page_count": len(pages)
        }
    )

    return OCRResponse(
        text=lc_document.page_content,
        pages=[p.model_dump() for p in pages_data]
    )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ocr_engine": "EasyOCR",
        "languages": ["th", "en"],
        "gpu": False
    }

@app.get("/")
async def root():
    return {
        "message": "OCR API running",
        "endpoint": "/ocr/pdf"
    }
