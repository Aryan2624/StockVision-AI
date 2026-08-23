import threading
from datetime import datetime

from .multi_model_manager import (
    normalize_symbol,
    multi_model_exists,
    validate_stock,
    train_multi_model,
)


training_jobs = {}
jobs_lock = threading.Lock()


def current_time():
    return datetime.now().isoformat(
        timespec="seconds"
    )


def get_training_status(symbol):
    symbol = normalize_symbol(symbol)

    # Model already available
    if multi_model_exists(symbol):
        return {
            "symbol": symbol,
            "status": "ready",
            "stage": "Ready",
            "message": "Existing multi-horizon model is ready.",
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    with jobs_lock:
        job = training_jobs.get(symbol)

        if job:
            return job.copy()

    return {
        "symbol": symbol,
        "status": "idle",
        "stage": "Waiting",
        "message": "Training has not started.",
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


def update_job(symbol, **values):
    with jobs_lock:
        if symbol not in training_jobs:
            training_jobs[symbol] = {}

        training_jobs[symbol].update(values)


def training_worker(symbol):
    try:
        # -----------------------------------------
        # VALIDATION
        # -----------------------------------------

        update_job(
            symbol,
            status="training",
            stage="Validating stock",
            message="Checking stock and historical market data.",
            error=None,
        )

        validate_stock(symbol)

        # -----------------------------------------
        # TRAINING
        # -----------------------------------------

        update_job(
            symbol,
            status="training",
            stage="Training BiLSTM",
            message=(
                "Training 1D, 3D, 5D and "
                "10D Multi-Horizon BiLSTM."
            ),
        )

        train_multi_model(symbol)

        # -----------------------------------------
        # VERIFY MODEL
        # -----------------------------------------

        update_job(
            symbol,
            status="training",
            stage="Evaluating model",
            message="Checking trained model and evaluation files.",
        )

        if not multi_model_exists(symbol):
            raise RuntimeError(
                "Training finished but model files were not created."
            )

        # -----------------------------------------
        # READY
        # -----------------------------------------

        update_job(
            symbol,
            status="ready",
            stage="Ready",
            message="Model trained and ready for forecasting.",
            finished_at=current_time(),
            error=None,
        )

    except Exception as e:
        update_job(
            symbol,
            status="error",
            stage="Training failed",
            message="Model training failed.",
            finished_at=current_time(),
            error=str(e),
        )


def start_training(symbol):
    symbol = normalize_symbol(symbol)

    # -----------------------------------------
    # ALREADY TRAINED
    # -----------------------------------------

    if multi_model_exists(symbol):
        return {
            "symbol": symbol,
            "status": "ready",
            "stage": "Ready",
            "message": "Existing multi-horizon model is already available.",
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    # -----------------------------------------
    # CHECK EXISTING TRAINING
    # -----------------------------------------

    with jobs_lock:
        existing_job = training_jobs.get(symbol)

        if (
            existing_job
            and existing_job.get("status") == "training"
        ):
            return existing_job.copy()

        training_jobs[symbol] = {
            "symbol": symbol,
            "status": "training",
            "stage": "Starting",
            "message": "Starting background model training.",
            "started_at": current_time(),
            "finished_at": None,
            "error": None,
        }

    # -----------------------------------------
    # BACKGROUND THREAD
    # -----------------------------------------

    thread = threading.Thread(
        target=training_worker,
        args=(symbol,),
        daemon=True,
    )

    thread.start()

    return get_training_status(symbol)