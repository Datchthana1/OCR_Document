from pdf2image import convert_from_bytes
import easyocr
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import cv2

LAYOUT_ANALYSIS_AVAILABLE = False
layout_engine = None

try:
    from paddleocr import PaddleOCR
    layout_engine = "paddleocr"
    LAYOUT_ANALYSIS_AVAILABLE = True
    print("PaddleOCR is available - using PaddleOCR for layout analysis")
except ImportError:
    try:
        import layoutparser as lp
        layout_engine = "layoutparser"
        LAYOUT_ANALYSIS_AVAILABLE = True
        print("LayoutParser is available - using LayoutParser for layout analysis")
    except ImportError as e:
        print(f"No layout analysis engine available: {e}")
        print("Running in OCR-only mode.")


app = FastAPI(title="OCR API EasyOCR with Layout Analysis", description="PDF OCR Service with Thai/English support and Layout Analysis")

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class LayoutRegion(BaseModel):
    type: str
    bbox: BoundingBox
    text: str
    confidence: float

class PageContent(BaseModel):
    page_number: int
    content: str
    layout_regions: Optional[List[LayoutRegion]] = None

class OCRResponse(BaseModel):
    pages: List[PageContent]
    full_text: str

layout_model = None
if LAYOUT_ANALYSIS_AVAILABLE:
    try:
        print("Loading Layout Analysis model...")
        if layout_engine == "paddleocr":
            layout_model = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                structure_version='PP-StructureV2'
            )
            print("PaddleOCR layout model loaded successfully")
        elif layout_engine == "layoutparser":
            layout_model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            print("LayoutParser model loaded successfully")
    except Exception as e:
        print(f"Failed to load layout model: {e}")
        print("Continuing without layout analysis")
        LAYOUT_ANALYSIS_AVAILABLE = False

print("Loading EasyOCR...")
reader = easyocr.Reader(['th', 'en'], gpu=False)
print("EasyOCR loaded successfully")


def detect_layout(image: np.ndarray):
    if not LAYOUT_ANALYSIS_AVAILABLE or layout_model is None:
        return None

    try:
        if layout_engine == "paddleocr":
            result = layout_model.ocr(image, cls=True)
            if not result or not result[0]:
                return None

            regions = []
            for line in result[0]:
                bbox = line[0]
                text_info = line[1]

                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                region = {
                    'type': 'Text',
                    'coordinates': (x1, y1, x2, y2),
                    'text': text_info[0],
                    'confidence': text_info[1]
                }
                regions.append(region)

            regions.sort(key=lambda r: (r['coordinates'][1], r['coordinates'][0]))
            return regions

        elif layout_engine == "layoutparser":
            layout = layout_model.detect(image)
            layout = lp.Layout([b for b in layout])
            sorted_layout = layout.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
            return sorted_layout

    except Exception as e:
        print(f"Error in layout detection: {e}")
        return None


def extract_text_from_region(image: np.ndarray, bbox: tuple, reader) -> tuple:
    x1, y1, x2, y2 = map(int, bbox)
    padding = 5
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return "", 0.0

    result = reader.readtext(cropped)

    if not result:
        return "", 0.0
    texts = []
    confidences = []
    for _, text, conf in result:
        texts.append(text)
        confidences.append(conf)

    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return combined_text, avg_confidence


@app.post("/ocr/pdf", response_model=OCRResponse)
async def pdf_ocr(request: Request):
    if not LAYOUT_ANALYSIS_AVAILABLE:
        print("Layout analysis not available, using simple OCR mode")
        return await pdf_ocr_simple(request)

    try:
        pdf_bytes = await request.body()

        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="No data received")

        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="Invalid PDF format")

        pages = convert_from_bytes(pdf_bytes, dpi=400, poppler_path='/opt/homebrew/bin')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    all_text = []
    pages_data = []

    for i, page in enumerate(pages):
        print(f"Processing page {i+1}...")

        page_array = np.array(page)

        layout = detect_layout(page_array)

        if layout is None or len(layout) == 0:
            print(f"Layout detection failed for page {i+1}, using simple OCR")
            result = reader.readtext(page_array)
            page_text = " ".join([text for _, text, _ in result])
            all_text.append(page_text)
            pages_data.append(PageContent(page_number=i+1, content=page_text))
            continue

        layout_regions = []
        page_text_parts = []

        for block in layout:
            if layout_engine == "paddleocr":
                region_type = block.get('type', 'Text')
                bbox = block['coordinates']
                text = block['text']
                confidence = block['confidence']
            else:
                region_type = block.type
                bbox = block.coordinates
                text, confidence = extract_text_from_region(page_array, bbox, reader)

            if text.strip():
                layout_regions.append(LayoutRegion(
                    type=region_type,
                    bbox=BoundingBox(
                        x1=float(bbox[0]),
                        y1=float(bbox[1]),
                        x2=float(bbox[2]),
                        y2=float(bbox[3])
                    ),
                    text=text,
                    confidence=confidence
                ))

                if region_type == "Title":
                    page_text_parts.append(f"\n=== {text} ===\n")
                elif region_type == "Table":
                    page_text_parts.append(f"\n[TABLE]\n{text}\n[/TABLE]\n")
                elif region_type == "List":
                    page_text_parts.append(f"\n{text}\n")
                elif region_type == "Figure":
                    page_text_parts.append(f"\n[FIGURE: {text}]\n")
                else:
                    page_text_parts.append(text)

        page_text = " ".join(page_text_parts)
        all_text.append(page_text)

        pages_data.append(PageContent(
            page_number=i+1,
            content=page_text,
            layout_regions=layout_regions
        ))

        print(f"--- Page {i+1} ---")
        print(f"Found {len(layout_regions)} layout regions")
        for region in layout_regions:
            print(f"  [{region.type}] {region.text[:100]}...")
        print("\n")

    return OCRResponse(
        pages=pages_data,
        full_text="\n\n".join(all_text)
    )


@app.post("/ocr/pdf/simple", response_model=OCRResponse)
async def pdf_ocr_simple(request: Request):
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
    return {
        "message": "OCR API with Layout Analysis is running",
        "endpoints": [
            "/ocr/pdf - OCR with layout analysis",
            "/ocr/pdf/simple - Simple OCR without layout analysis"
        ]
    }


@app.get("/health")
async def health():
    models = ["EasyOCR (th, en)"]
    layout_info = {}

    if LAYOUT_ANALYSIS_AVAILABLE and layout_model is not None:
        if layout_engine == "paddleocr":
            models.append("PaddleOCR (Layout Analysis)")
            layout_info = {"engine": "PaddleOCR", "version": "PP-StructureV2"}
        elif layout_engine == "layoutparser":
            models.append("LayoutParser (PubLayNet)")
            layout_info = {"engine": "LayoutParser", "model": "Detectron2"}

    return {
        "status": "healthy",
        "models": models,
        "layout_analysis_enabled": LAYOUT_ANALYSIS_AVAILABLE and layout_model is not None,
        "layout_engine": layout_info
    }
