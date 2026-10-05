import asyncio
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_single_run")

import httpx
from app.assistant.context_builder import run_assistant_pipeline
from app.llm.meteorologist import run_meteorologist
from app.llm.writer import run_writer
from app.schemas import AssistantRequest, ReportParams

async def main():
    print("=" * 60)
    print("RUNNING SINGLE VERIFICATION PIPELINE TEST")
    print("Location: Mumbai, India")
    print("=" * 60)

    # 1. Block 1: Assistant
    req = AssistantRequest(
        query="Mumbai",
        latitude=19.0760,
        longitude=72.8777,
        context_style="hierarchical",
        use_cache=False,  # Run fresh once
    )

    async with httpx.AsyncClient(verify=False) as client:
        print("\n[Stage 1/3] Running Assistant Data Pipeline...")
        payload, meta = await run_assistant_pipeline(req, client)
        print(f"-> Resolved Location: {payload.location.city}, {payload.location.country}")
        print(f"-> Elevation: {payload.location.elevation_m}m")
        print(f"-> Hourly records fetched: {len(payload.hourly)}")
        print(f"-> 6-Hour aggregate windows: {len(payload.six_hour_aggregates)}")
        print(f"-> Daily aggregate rows: {len(payload.daily_aggregates)}")
        print(f"-> Climatology monthly records: {len(payload.climatology.monthly) if payload.climatology else 0}")

    # 2. Block 2: Meteorologist + Verification Engine
    print("\n[Stage 2/3] Calling Gemini for Meteorologist Reasoning & Executing Verification Engine...")
    met_output, met_meta = await run_meteorologist(payload, use_cache=False)
    
    print("\n--- METEOROLOGIST OUTPUT ---")
    print(f"Summary:\n{met_output.summary}\n")
    print(f"Proof:\n{met_output.proof}\n")
    print(f"Keywords: {met_output.keywords}")
    print(f"Warnings: {met_output.warnings or 'None'}")
    print(f"Causal Chain: {met_output.causal_chain}")
    print(f"Claims Count: {len(met_output.claims)}")

    conf = met_output.confidence_report or {}
    print("\n" + "=" * 60)
    print("VERIFICATION ENGINE ACCURACY AUDIT")
    print("=" * 60)
    print(f"Overall Confidence Score: {conf.get('overall_score', 0.0):.4f}")
    scores = conf.get("scores", {})
    print(f"  - Data Fact Pass Score:         {scores.get('data_fact', 0.0):.4f}")
    print(f"  - Meteorological Rule Score:    {scores.get('met_rule', 0.0):.4f}")
    print(f"  - Causal Graph Validity:        {scores.get('causal_graph', 0.0):.4f}")
    print(f"  - Temporal Consistency:         {scores.get('temporal_consistency', 0.0):.4f}")

    failures = conf.get("failures", [])
    if failures:
        print(f"\nDetected Failures/Warnings ({len(failures)}):")
        for idx, f in enumerate(failures, 1):
            print(f"  {idx}. [{f.get('severity', '').upper()}] [{f.get('component', '')}] {f.get('message', '')}")
    else:
        print("\nAll verification checks passed with ZERO failures!")

    # 3. Block 3: Writer
    print("\n[Stage 3/3] Calling Writer Agent for Report Adaptation...")
    params = ReportParams(tone="conversational", length="medium", domain="general_public")
    report, report_meta = await run_writer(payload, met_output, params, use_cache=False)

    print("\n--- FINAL WRITER REPORT ---")
    print(f"Title: {report.header.title}")
    print(f"Info:  {report.header.information}")
    print(f"Final Summary:\n{report.analysis.summary}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY (1 LLM call cycle used)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
