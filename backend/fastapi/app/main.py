"""FastAPI main entry point for the NeuraSight ML Service.

Configures the application lifespan (model loading at startup),
registers all routers, initializes the AI engine module registry,
and adds a global exception handler that never exposes stack traces
to clients.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, predict, gradcam, report
from app.services import llm
from app.services.inference import load_model
from app.services.ensemble import load_ensemble
from app.services.knowledge import get_knowledge_base
from app.services.model_cache import get_device, model_cache
from app.services.narratives import get_narrative_store
from app.services.uncertainty import CalibrationBundle
from app.services.validation import OODDetector
from app.engine.registry import init_registry, get_registry

# Configure the root logger so this package's log calls are actually visible.
# Without this, every logger.info in the app was silently discarded: preload
# success, calibration state and unfaithful-explanation warnings were all
# invisible, leaving latency as the only observable signal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    On startup:
        1. Initialize the module registry (discovers brain_mri, chest_xray, etc.)
        2. Load the stacking ensemble (4 base models + meta-learner) into app.state.ensemble
        3. Load EfficientNet-B0 as app.state.model for single-model fallback
    On shutdown: performs cleanup if needed.
    """
    # Startup
    app.state.ensemble = None
    app.state.model = None

    # Initialize module registry
    init_registry()
    registry = get_registry()
    app.state.registry = registry
    logger.info("Module registry initialized: %s", registry.list_ids())

    device = get_device()
    app.state.device = str(device)
    app.state.enforce_quality_gate = settings.ENFORCE_QUALITY_GATE

    # Load the decision-support artefacts once and report what is actually
    # usable. Each layer degrades independently: missing calibration means raw
    # probabilities, missing narratives means template text. None of them should
    # prevent the service from starting.
    app.state.calibration = CalibrationBundle.load(settings.CALIBRATION_DIR)
    app.state.ood_detector = OODDetector.load(settings.CALIBRATION_DIR)
    app.state.knowledge = get_knowledge_base()
    app.state.narratives = get_narrative_store()

    logger.info(
        "Calibration: available=%s design=%s T=%.4f conformal_alpha=%s q_hat=%s",
        app.state.calibration.available,
        app.state.calibration.design,
        app.state.calibration.temperature,
        app.state.calibration.alpha,
        app.state.calibration.q_hat,
    )
    logger.info(
        "Input validation: available=%s detector=%s threshold=%s gate_enforced=%s",
        app.state.ood_detector.available,
        app.state.ood_detector.primary_detector,
        app.state.ood_detector.threshold,
        settings.ENFORCE_QUALITY_GATE,
    )

    kb_stats = app.state.knowledge.stats()
    logger.info(
        "Knowledge base: %d entries, %d chunks, %d sources, %d unresolved citations",
        kb_stats["entries"],
        kb_stats["chunks"],
        kb_stats["sources_registered"],
        len(kb_stats["unresolved_citations"]),
    )

    narrative_stats = app.state.narratives.stats()
    logger.info(
        "Narratives: %d total, %d reviewed and servable, %d awaiting review",
        narrative_stats["total"],
        narrative_stats["reviewed_and_servable"],
        narrative_stats["awaiting_review"],
    )
    if narrative_stats["total"] and not narrative_stats["reviewed_and_servable"]:
        logger.warning(
            "No narrative is approved, so all reports will use template text."
        )

    if not llm.is_configured():
        logger.info(
            "No LLM key configured. This is fine at request time: narratives are "
            "authored offline and only the vetted text is served."
        )

    # Preload base models so weights are read from disk once per process
    # instead of once per request. Which modules are eager is configurable;
    # anything omitted is still served, just cached on first use.
    specs: list[dict] = []
    for module_id in settings.preload_modules:
        mod = registry.get(module_id)
        if mod is None:
            logger.warning("PRELOAD_MODULES lists unknown module '%s'", module_id)
            continue
        if not mod.is_available():
            logger.warning("Skipping preload for '%s': weights missing", module_id)
            continue
        specs.extend(mod.preload_specs())

    if specs:
        result = model_cache.preload(specs)
        logger.info(
            "Preloaded %d/%d models on %s: %s",
            len(result["loaded"]),
            len(specs),
            device,
            result["loaded"],
        )
        if result["failed"]:
            logger.error("Model preload failures: %s", result["failed"])

    if settings.USE_ENSEMBLE:
        try:
            ensemble = load_ensemble(settings.MODELS_DIR)
            app.state.ensemble = ensemble
            logger.info(
                "Ensemble config loaded from %s | models=%s | meta=%s | features=%s",
                settings.MODELS_DIR,
                ensemble["order"],
                ensemble["config"].get("meta_learner"),
                getattr(ensemble["meta"], "n_features_in_", "?"),
            )
        except Exception as e:
            logger.error("Failed to load ensemble from %s: %s", settings.MODELS_DIR, e)

    # A usable single model is the minimum bar for serving brain MRI requests.
    try:
        app.state.model = load_model(settings.MODEL_PATH)
        logger.info("Single-model fallback ready from %s", settings.MODEL_PATH)
    except Exception as e:
        # Do not kill the process: another module (e.g. chest_xray) may have
        # perfectly good weights and can still serve traffic. /health reports
        # per-module readiness so callers can tell what is actually usable.
        logger.error(
            "Brain MRI fallback model unavailable (%s: %s). Brain requests will "
            "fail; other modules remain available.",
            type(e).__name__,
            e,
        )

    yield
    # Shutdown
    logger.info("Shutting down NeuraSight ML Service")


app = FastAPI(title="NeuraSight ML Service", lifespan=lifespan)

# CORS — allow browser requests from Vercel and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(gradcam.router)
app.include_router(report.router)


@app.get("/modules")
async def get_modules():
    """Return the list of available AI modules.

    Each module includes its ID, display name, supported classes,
    and whether model weights are currently available.
    """
    from app.engine.registry import get_registry

    registry = get_registry()
    return {"modules": registry.available_modules()}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return a generic 500 error.

    Logs the full exception details for debugging but never exposes
    stack traces, file paths, or environment variables to the client.
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred"},
    )
