"""
RapidOCR sidecar for CV text extraction.
Accepts a PDF (base64) and returns extracted text.
Deploy on Render free tier (512 MB RAM, cold start ~30-60s).
Memory-optimized: lower DPI, page-by-page, explicit cleanup.
"""
import base64
import gc
import io
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # pymupdf
from rapidocr_onnxruntime import RapidOCR

app = FastAPI(title="OCR Service", version="1.1")
ocr = None

def get_ocr():
    global ocr
    if ocr is None:
        ocr = RapidOCR()
    return ocr

class OcrRequest(BaseModel):
    pdf_base64: str
    filename: str = "document.pdf"

class OcrResponse(BaseModel):
    text: str
    pages: int
    method: str
    elapsed_ms: int

@app.get("/health")
def health():
    return {"status": "ok", "service": "rapidocr"}

@app.post("/ocr", response_model=OcrResponse)
def ocr_pdf(req: OcrRequest):
    t0 = time.time()
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base64: {e}")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad pdf: {e}")
    engine = get_ocr()
    all_text = []
    page_count = len(doc)
    for i in range(page_count):
        try:
            page = doc[i]
            # 1.5x (108 DPI) — lower than 2x to save memory on 512MB tier
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
            del pix
            result, _ = engine(img_bytes)
            del img_bytes
            if result:
                page_text = "\n".join([line[1] for line in result])
                all_text.append(page_text)
            del result
        except Exception as e:
            all_text.append(f"[OCR error on page {i+1}: {e}]")
        gc.collect()
    doc.close()
    del doc
    gc.collect()
    text = "\n\n".join(all_text).strip()
    elapsed = int((time.time() - t0) * 1000)
    return OcrResponse(text=text, pages=len(all_text), method="rapidocr", elapsed_ms=elapsed)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
