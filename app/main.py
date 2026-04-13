from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from app.assistant.context_builder import run_assistant_pipeline
from app.config import get_settings
from app.jsonutil import canonical_json
from app.llm.meteorologist import run_meteorologist
from app.llm.writer import run_writer
from app.schemas import (
    AnalysisRequest,
    AssistantRequest,
    FullPipelineRequest,
    ReportParams,
    ReportRequest,
)
from app.validation import validate_context_dict

def redact_secrets(text: str) -> str:
    """Redact sensitive API keys and usernames from strings."""
    try:
        from app.config import get_settings
        s = get_settings()
        sensitive = [s.gemini_api_key, s.geonames_username, s.cds_key]
        for val in sensitive:
            if val and len(str(val)) > 3:
                text = text.replace(str(val), "********")
    except Exception:
        pass
    return text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aimeteo")

_ROOT = Path(__file__).resolve().parent.parent
_UI_DIR = _ROOT / "ui"

app = FastAPI(title="Hierarchical AI-Meteorologist", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    get_settings()
    logger.info("Startup complete")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def serve_ui() -> FileResponse:
    """Demo UI: data → aggregation → meteorologist → writer."""
    index = _UI_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found. Expected ui/index.html")
    return FileResponse(index)


@app.post("/api/pipeline")
async def post_full_pipeline(req: FullPipelineRequest) -> JSONResponse:
    """
    Full chain in one request: Assistant → Meteorologist → Writer.
    Returns context, raw meteorologist JSON, and final report for the demo UI.
    """
    ar = AssistantRequest(
        query=req.query,
        latitude=req.latitude,
        longitude=req.longitude,
        context_style=req.context_style,
        use_cache=req.use_cache,
    )
    async with httpx.AsyncClient() as client:
        try:
            payload, assistant_meta = await run_assistant_pipeline(ar, client)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=redact_secrets(str(e))) from e
        except Exception as e:
            logger.exception("Assistant pipeline failed: %s", redact_secrets(str(e)))
            raise HTTPException(status_code=502, detail=f"Assistant pipeline failed: {redact_secrets(str(e))}") from e

    try:
        met, m_meta = await run_meteorologist(payload, use_cache=req.use_cache)
        params = ReportParams(tone=req.tone, length=req.length, domain=req.domain)
        report, w_meta = await run_writer(payload, met, params, use_cache=req.use_cache)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline LLM stages failed: %s", redact_secrets(str(e)))
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {redact_secrets(str(e))}") from e

    deg = list(
        dict.fromkeys(
            payload.degradation_codes
            + m_meta.get("degradation", [])
            + w_meta.get("degradation", [])
        )
    )
    return JSONResponse(
        content={
            "context": payload.model_dump(mode="json"),
            "assistant_meta": assistant_meta,
            "meteorologist": met.model_dump(mode="json"),
            "meteorologist_meta": m_meta,
            "report": report.model_dump(mode="json", exclude_none=True),
            "report_meta": w_meta,
            "degradation_codes": deg,
            "context_frozen": canonical_json(payload.model_dump(mode="json")),
        }
    )


@app.post("/context")
async def post_context(req: AssistantRequest) -> JSONResponse:
    """Run Block 1 (Assistant): resolve location, fetch data, build serialized context payload."""
    async with httpx.AsyncClient() as client:
        try:
            payload, meta = await run_assistant_pipeline(req, client)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=redact_secrets(str(e))) from e
        except Exception as e:
            logger.exception("Assistant pipeline failed: %s", redact_secrets(str(e)))
            raise HTTPException(status_code=502, detail=f"Assistant pipeline failed: {redact_secrets(str(e))}") from e

    body: dict[str, Any] = {
        "context": payload.model_dump(mode="json"),
        "assistant_meta": meta,
        "context_frozen": canonical_json(payload.model_dump(mode="json")),
        "degradation_codes": payload.degradation_codes,
    }
    return JSONResponse(content=body)


@app.post("/analysis")
async def post_analysis(req: AnalysisRequest) -> JSONResponse:
    """Run Block 2 (Meteorologist) only."""
    ctx = validate_context_dict(req.context)
    try:
        out, meta = await run_meteorologist(ctx, use_cache=req.use_cache)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Meteorologist failed: %s", redact_secrets(str(e)))
        raise HTTPException(status_code=502, detail=f"Meteorologist failed: {redact_secrets(str(e))}") from e

    deg = list(dict.fromkeys(ctx.degradation_codes + meta.get("degradation", [])))
    return JSONResponse(
        content={
            "summary": out.summary,
            "proof": out.proof,
            "keywords": out.keywords,
            "warnings": out.warnings,
            "degradation_codes": deg,
            "cached": meta.get("cached", False),
            "cache_key": meta.get("cache_key"),
            "model": get_settings().gemini_model,
        }
    )


@app.post("/report")
async def post_report(req: ReportRequest) -> JSONResponse:
    """Run Block 2 then Block 3 (full pipeline)."""
    ctx = validate_context_dict(req.context)
    params = ReportParams(tone=req.tone, length=req.length, domain=req.domain)
    try:
        met, m_meta = await run_meteorologist(ctx, use_cache=req.use_cache)
        report, w_meta = await run_writer(ctx, met, params, use_cache=req.use_cache)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Report pipeline failed: %s", redact_secrets(str(e)))
        raise HTTPException(status_code=502, detail=f"Report pipeline failed: {redact_secrets(str(e))}") from e

    deg = list(
        dict.fromkeys(
            ctx.degradation_codes + m_meta.get("degradation", []) + w_meta.get("degradation", [])
        )
    )
    logger.info("Report emitted context_mode=%s", report.log_context_mode)
    return JSONResponse(
        content={
            "report": report.model_dump(mode="json", exclude_none=True),
            "degradation_codes": deg,
            "cached_meteorologist": m_meta.get("cached", False),
            "cached_report": w_meta.get("cached", False),
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
