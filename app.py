"""
RapidOCR sidecar for CV text extraction.
Accepts a PDF (base64) and returns extracted text.
Deploy on Render free tier (512 MB RAM, cold start ~30-60s).
Memory-optimized v1.2: lower DPI, page-by-page, explicit cleanup, error handling.
"""
import base64
import gc
import io
import os
import time
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # pymupdf
from rapidocr_onnxruntime import RapidOCR

app = FastAPI(title="OCR Service", version="1.2")
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
    return {"status": "ok", "service": "rapidocr", "version": "1.2"}

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
    try:
        engine = get_ocr()
        all_text = []
        page_count = len(doc)
        for i in range(page_count):
            try:
                page = doc[i]
                # 1.0x (72 DPI) — minimum for 512MB tier
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)[:500]}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
