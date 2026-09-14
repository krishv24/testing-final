-- =============================================================
-- Supabase Setup for IEEE Experiment Data Collection
-- Run this ONCE in the Supabase SQL Editor:
--   https://supabase.com/dashboard → SQL Editor → New Query
-- =============================================================

CREATE TABLE IF NOT EXISTS experiment_runs (
    -- Identity
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now() NOT NULL,

    -- Who ran it
    runner_name     TEXT        NOT NULL DEFAULT 'unknown',

    -- Location
    location_name   TEXT        NOT NULL,
    latitude        FLOAT8,
    longitude       FLOAT8,
    climate_zone    TEXT,

    -- Pipeline parameters
    context_style   TEXT        DEFAULT 'hierarchical',
    tone            TEXT,
    domain          TEXT,
    length          TEXT        DEFAULT 'short',
    gemini_model    TEXT,

    -- ── Key Scores (flat columns for easy SQL queries / CSV export) ──
    overall_score           FLOAT8,
    fact_score              FLOAT8,
    rule_score              FLOAT8,
    causal_score            FLOAT8,
    temporal_score          FLOAT8,

    -- ── Evaluation Metrics ──
    false_claim_rate        FLOAT8,
    total_claims            INT4,
    failed_claims           INT4,
    rule_violations         INT4,
    verified_report         BOOL,
    cross_scale_consistent  BOOL,
    correction_needed       BOOL,
    hallucination_precision FLOAT8,
    hallucination_recall    FLOAT8,

    -- ── Timing (milliseconds) ──
    total_pipeline_ms       INT4,
    assistant_ms            INT4,
    meteorologist_ms        INT4,
    writer_ms               INT4,

    -- ── Rich Data (JSONB — full LLM outputs & validation details) ──
    meteorologist_raw           JSONB,
    writer_raw                  JSONB,
    fact_validation_details     JSONB,
    rule_validation_details     JSONB,
    causal_validation_details   JSONB,
    temporal_validation_details JSONB,
    confidence_failures         JSONB,
    degradation_codes           JSONB,

    -- ── Error tracking (NULL = success) ──
    error TEXT
);

-- Disable Row Level Security so all team members can read/write with the publishable key
ALTER TABLE experiment_runs DISABLE ROW LEVEL SECURITY;

-- Grant full access to the anon and authenticated roles (required for Supabase API keys)
GRANT ALL ON experiment_runs TO anon;
GRANT ALL ON experiment_runs TO authenticated;

-- ── Indexes for common queries ──
CREATE INDEX IF NOT EXISTS idx_runs_location   ON experiment_runs (location_name);
CREATE INDEX IF NOT EXISTS idx_runs_runner     ON experiment_runs (runner_name);
CREATE INDEX IF NOT EXISTS idx_runs_score      ON experiment_runs (overall_score);
CREATE INDEX IF NOT EXISTS idx_runs_successful ON experiment_runs (location_name, tone, domain) WHERE error IS NULL;

-- ── Helpful view: only successful runs ──
CREATE OR REPLACE VIEW experiment_runs_success AS
SELECT * FROM experiment_runs WHERE error IS NULL;
