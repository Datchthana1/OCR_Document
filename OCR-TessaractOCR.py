from pdf2image import convert_from_bytes
import pytesseract
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from img2table.ocr import TesseractOCR
from img2table.document import Image as Img2TableImage
import io
import re


app = FastAPI(title="OCR API Tesseract", description="PDF OCR Service with Thai and English support")

def clean_ocr_text(text: str) -> str:
    noise_patterns = [
        r'\b[A-Z]\s+[A-Z]\s+[A-Z]',
        r'\b[A-Z]{1,2}\b(?=\s+[A-Z]{1,2}\b)',
        r'\b(wv|eo|wo|wa|KK|K)\b',
        r'[ฆๅ]+\s*[๐-๙]+',
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    allowed_pattern = r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,!?;:()\[\]{}\'\"@#$%&*+\-=/<>₿฿]'
    cleaned = re.sub(allowed_pattern, '', text)

    cleaned = ' '.join(cleaned.split())

    cleaned = re.sub(r'\s+\b[a-zA-Z]{1,2}\b\s+', ' ', cleaned)

    return cleaned.strip()

class TableData(BaseModel):
    table_number: int
    data: List[List[str]]
    html: Optional[str] = None

class PageContent(BaseModel):
    page_number: int
    content: str
    tables: List[TableData] = []

class OCRResponse(BaseModel):
    pages: List[PageContent]
    full_text: str
    total_tables: int = 0

@app.post("/ocr/pdf", response_model=OCRResponse)
async def pdf_ocr(request: Request, extract_tables: bool = True):
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

    tesseract_config = r'--oem 1 --psm 12 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์ํ๎abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]{}@#$%&*+-=/<>฿"'

    ocr = TesseractOCR(lang='tha+eng')

    all_text = []
    pages_data = []
    total_tables = 0

    for i, page in enumerate(pages):
        page_text = pytesseract.image_to_string(
            page,
            lang='tha+eng',
            config=tesseract_config
        )

        page_text_cleaned = clean_ocr_text(page_text)

        all_text.append(page_text_cleaned)

        tables_list = []

        if extract_tables:
            try:
                img_byte_arr = io.BytesIO()
                page.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                img2table_doc = Img2TableImage(src=img_byte_arr)
                extracted_tables = img2table_doc.extract_tables(ocr=ocr)

                for table_idx, table in enumerate(extracted_tables):
                    df = table.df

                    df = df.applymap(lambda x: clean_ocr_text(str(x)) if x else '')

                    table_data = df.values.tolist()
                    table_html = df.to_html(index=False, border=1)

                    tables_list.append(TableData(
                        table_number=table_idx + 1,
                        data=table_data,
                        html=table_html
                    ))
                    total_tables += 1

            except Exception as e:
                print(f"Table extraction error on page {i+1}: {str(e)}")

        pages_data.append(PageContent(
            page_number=i+1,
            content=page_text_cleaned,
            tables=tables_list
        ))

        print(f"--- Page {i+1} ---")
        print(page_text_cleaned)
        if tables_list:
            print(f"Found {len(tables_list)} table(s)")
        print("\n")

    return OCRResponse(
        pages=pages_data,
        full_text=" ".join(all_text),
        total_tables=total_tables
    )

@app.get("/")
async def root():
    return {
        "message": "OCR API is running",
        "endpoints": [
            "/ocr/pdf - POST PDF file, supports table extraction",
            "/health - GET health check"
        ],
        "features": [
            "Thai and English OCR",
            "Table extraction (img2table)",
            "Per-page content and tables"
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
