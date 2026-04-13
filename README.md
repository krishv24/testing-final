# Hierarchical AI-Meteorologist

Status: Production-Ready
Architecture: Multi-Agent LLM Pipeline
Domain: Explainable Meteorological Diagnostics

The Hierarchical AI-Meteorologist is a sophisticated research-driven framework designed to provide explainable weather diagnostics. Unlike standard weather apps, this system employs a hierarchical reasoning pipeline that mimics human meteorological analysis by aggregating data across multiple temporal scales before generating natural language narratives.

## Core Workflow Implementation

The system utilizes a structured three-phase pipeline to ensure factual accuracy and logical consistency:

1.  **Orchestration and Data Retrieval**:
    *   **Geospatial Processing**: Resolves location queries into coordinate-level metadata, including elevation and regional administrative tags.
    *   **Multi-Source Harvesting**: Aggregates high-resolution forecast data from Open-Meteo and 30-year climatological normals (1991-2020) for historical context.
    *   **Atmospheric Discussion**: Integrates expert human-written discussions (NWS AFD) for local meteorological nuance where available.

2.  **Hierarchical Temporal Aggregation**:
    *   Processes raw high-frequency hourly data into a structured hierarchy of 6-hour and daily aggregates.
    *   Applies statistical normalization and circular wind analysis to provide the LLM with "pre-reasoned" data points, reducing hallucination risks.
    *   Employs a dynamic context strategy that adjusts the granularity of data (Hourly vs. 6-Hour) based on the forecast lead time to optimize token consumption and reasoning depth.

3.  **Bimodal Reasoning and Synthesis**:
    *   **The Meteorologist Agent**: Performs deterministic atmospheric analysis. It generates a "Proof" block—a verification layer that maps every narrative assertion to specific data signals in the aggregates.
    *   **The Writer Agent**: Transmutes technical diagnostics into domain-specific reports. It adapts the output according to user-defined parameters such as audience domain (Risk, Energy, Public), tone, and technical depth.

## System Architecture

The pipeline is organized into three distinct execution modules:

### Block 1: The Assistant (Orchestrator)
Handles the non-deterministic data layer. It manages geocoding, climatology normalization, and the hierarchical compression of hourly signals.

### Block 2: The Meteorologist (Reasoning Engine)
A specialized LLM agent that synthesizes atmospheric dynamics. It identifies keywords, detects anomalies compared to historical norms, and provides evidence-based diagnostics in structured JSON.

### Block 3: The Writer (Communication Layer)
A context-aware formatting agent. It maintains strict fidelity to the Meteorologist's findings while adjusting the presentation for specific professional domains or public safety requirements.

---

## Technical Specifications

### API Infrastructure
The backend is built on FastAPI, providing high-performance asynchronous endpoints:

*   **POST /api/pipeline**: The primary entry point for a one-shot execution of the entire three-block sequence.
*   **POST /context**: Executes Block 1, returning a serialized hierarchical context payload.
*   **POST /analysis**: Executes Block 2 for standalone meteorological reasoning.
*   **POST /report**: Executes the full sequence with customizable writer parameters.

### Technology Stack
*   **Core Logic**: Python 3.10+, FastAPI, Pydantic (V2)
*   **Intelligence Layer**: Google Gemini 2.0/2.5 series
*   **Analytical Engines**: Pandas, NumPy
*   **Data Sources**: Open-Meteo API, Meteostat, GeoNames, NOAA/NWS

---

## Configuration

The system requires a `.env` file containing the following variables:

*   **GEMINI_API_KEY**: Required for core reasoning and synthesis agents.
*   **GEONAMES_USERNAME**: Required for geospatial query resolution.
*   **CDS_KEY / CDS_URL**: (Optional) For high-fidelity ERA5 reanalysis fallback.
*   **CACHE_DIR**: Local path for result persistence (default: .cache).

---

## Installation and Deployment

For local execution:

```bash
# Initialize dependencies
pip install -r requirements.txt

# Launch the asynchronous server
python -m uvicorn app.main:app --reload
```

---
*Developed for the Hierarchical AI-Meteorologist Pipeline research.*
