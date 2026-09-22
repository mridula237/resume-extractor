import json
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from extract import extract

app = FastAPI(title="Resume Extractor")


@app.post("/extract")
async def extract_resume(
    file: UploadFile = File(...),
    model: str = Query(default="claude", enum=["claude", "gpt"]),
    cache: bool = Query(default=False),
    confidence: bool = Query(default=False),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = extract(tmp_path, model, cache, confidence)
    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "data": result["data"],
        "confidence": result.get("confidence", {}),
        "meta": {
            "model": result["model"],
            "input_tokens": result["in"],
            "output_tokens": result["out"],
            "latency_s": round(result["latency"], 2),
            "cache_read_tokens": result["cache_read"],
        }
    })


@app.get("/health")
def health():
    return {"status": "ok"}