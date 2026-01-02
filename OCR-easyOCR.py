from pdf2image import convert_from_bytes
import easyocr
import numpy as np
from typing import List
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel


app = FastAPI(title="OCR API EasyOCR", description="PDF OCR Service with Thai and English support")

class PageContent(BaseModel):
    page_number: int
    content: str

class OCRResponse(BaseModel):
    pages: List[PageContent]
    full_text: str

@app.post("/ocr/pdf", response_model=OCRResponse)
async def pdf_ocr(request: Request):
    try:
        pdf_bytes = await request.body()

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="No data received")

        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="Invalid PDF format")

        pages = convert_from_bytes(pdf_bytes, dpi=350, poppler_path='/opt/homebrew/bin')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    reader = easyocr.Reader(['th','en'], gpu=False)

    all_text = []
    pages_data = []

    for i, page in enumerate(pages):
        result = reader.readtext(np.array(page))
        page_text = " ".join([text for _, text, _ in result])
        all_text.append(page_text)
        pages_data.append(PageContent(page_number=i+1, content=page_text))
        print(f"--- Page {i+1} ---")
        print(page_text)
        print("\n")

    return OCRResponse(
        pages=pages_data,
        full_text="\n".join(all_text)
    )

@app.get("/")
async def root():
    return {"message": "OCR API is running", "endpoints": ["/ocr/pdf"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}
