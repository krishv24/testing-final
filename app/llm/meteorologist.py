from __future__ import annotations

import logging
import pathlib
import pandas as pd

from app.degradation import DegradationCode
from app.jsonutil import sha256_hex
from app.cache_store import cache_get_json, cache_set_json
from app.config import get_settings
from app.llm.gemini_client import generate_json_text, parse_json_object
from app.llm.prompts import meteorologist_prompt
from app.schemas import ContextPayload, MeteorologistOutput
from app.validation import validate_for_meteorologist

# Verification components
from app.verification.data_validator import DataFactValidator
from app.verification.rule_engine import MeteorologicalRuleEngine
from app.verification.causal_graph import CausalGraph
from app.verification.temporal_checker import TemporalConsistencyChecker
from app.verification.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


async def run_meteorologist(ctx: ContextPayload, *, use_cache: bool = True) -> tuple[MeteorologistOutput, dict[str, object]]:
    validate_for_meteorologist(ctx)
    settings = get_settings()
    key = sha256_hex(
        {"context": ctx.model_dump(mode="json"), "model": settings.gemini_model}
    )
    if use_cache:
        hit = cache_get_json("meteorologist", key)
        if hit and "output" in hit:
            out = MeteorologistOutput.model_validate(hit["output"])
            logger.info("Meteorologist cache hit key=%s", key[:16])
            return out, {"cached": True, "cache_key": key, "degradation": [DegradationCode.CACHE_HIT_METEOROLOGIST.value]}

    # Prepare dataframes for fact-checking
    hourly_df = pd.DataFrame([r.model_dump(mode="json") for r in ctx.hourly]) if ctx.hourly else pd.DataFrame()
    six_hour_df = pd.DataFrame([r.model_dump(mode="json") for r in ctx.six_hour_aggregates]) if ctx.six_hour_aggregates else pd.DataFrame()
    daily_df = pd.DataFrame([r.model_dump(mode="json") for r in ctx.daily_aggregates]) if ctx.daily_aggregates else pd.DataFrame()
    current_df = pd.DataFrame([ctx.current_conditions.model_dump(mode="json")]) if ctx.current_conditions else pd.DataFrame()

    # Paths to validator configurations
    verify_dir = pathlib.Path(__file__).parent.parent / "verification"
    thresholds_path = str(verify_dir / "thresholds.yaml")
    met_rules_path = str(verify_dir / "met_rules.yaml")
    weights_path = str(verify_dir / "verification_weights.yaml")

    # Instantiate validator components
    fact_validator = DataFactValidator(thresholds_path)
    rule_engine = MeteorologicalRuleEngine(met_rules_path)
    causal_graph = CausalGraph(met_rules_path)
    temporal_checker = TemporalConsistencyChecker()
    scorer = ConfidenceScorer(weights_path)

    prompt = meteorologist_prompt(ctx)
    current_prompt = prompt
    attempts = 0
    max_retry_attempts = 0  # Total of 1 run max (1 initial + 0 retries)
    final_report = None
    final_data = None

    while attempts <= max_retry_attempts:
        logger.info("Calling Gemini for Meteorologist (attempt %s/%s)", attempts + 1, max_retry_attempts + 1)
        raw = await generate_json_text(current_prompt, temperature=0.35)
        try:
            data = parse_json_object(raw)
        except Exception as e:
            logger.warning("Failed to parse JSON on attempt %s: %s", attempts + 1, e)
            if attempts == max_retry_attempts:
                raise
            current_prompt = f"{prompt}\n\nFAILED JSON PARSING: Please respond with raw JSON only, no markdown formatting or text around it."
            attempts += 1
            continue

        # Run validations
        claims = data.get("claims", [])
        causal_chain = data.get("causal_chain", [])

        fact_results = fact_validator.validate_all_claims(claims, hourly_df, six_hour_df, daily_df, current_df)
        rule_results = rule_engine.validate_rules(claims, causal_chain)
        causal_results = causal_graph.validate_chain(causal_chain)
        temporal_results = temporal_checker.validate_temporal_consistency(claims)

        # Scorer
        report = scorer.compute_report(fact_results, rule_results, causal_results, temporal_results)
        final_report = report
        final_data = data

        score = report["overall_score"]
        logger.info("Validation completed. Attempt %s Overall Score: %.2f (Required: %.2f)", attempts + 1, score, settings.verification_threshold)
        logger.info("Validation Failures: %s", report.get("failures", []))

        if score >= settings.verification_threshold or attempts == max_retry_attempts:
            break

        # Build feedback message for retry
        feedback_parts = [
            f"The previous response failed meteorological validation with a score of {score:.2f} (required: {settings.verification_threshold:.2f}).",
            "Please correct the narrative, proof, claims, and causal chain to resolve the following issues:"
        ]
        for idx, failure in enumerate(report["failures"], 1):
            feedback_parts.append(f"{idx}. [{failure['severity'].upper()}] Component: {failure['component']} - {failure['message']}")
        
        feedback_parts.append(
            "Ensure that all statements in the summary are strictly supported by numbers in the tables, "
            "and that the causal chain contains only physically possible, logically connected transitions."
        )
        feedback_message = "\n".join(feedback_parts)

        current_prompt = (
            f"{prompt}\n\n"
            f"--- FEEDBACK ON PREVIOUS ATTEMPT ---\n"
            f"{feedback_message}\n\n"
            f"Please revise your analysis to ensure all claims are completely correct, logically consistent, and match the facts."
        )
        attempts += 1

    # Attach confidence report to meteorologist output object
    out = MeteorologistOutput.model_validate(final_data)
    out.confidence_report = final_report

    cache_set_json(
        "meteorologist",
        key,
        {
            "output": out.model_dump(mode="json"),
            "context_hash": key,
        },
    )
    return out, {"cached": False, "cache_key": key, "degradation": []}
