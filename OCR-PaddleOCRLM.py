from fastapi import FastAPI, Request, HTTPException
from paddleocr import PaddleOCRVL
import tempfile
import os

app = FastAPI()

pipeline = PaddleOCRVL()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(request: Request):
    data: bytes = await request.body()

    if not data:
        raise HTTPException(status_code=400, detail="Empty body")

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        output = pipeline.predict(input=tmp_path)

        markdown_list = [res.markdown for res in output]
        markdown_text = pipeline.concatenate_markdown_pages(markdown_list)

        return {"markdown": markdown_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
