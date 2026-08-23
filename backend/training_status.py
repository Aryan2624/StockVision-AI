import threading
from datetime import datetime

from .multi_model_manager import (
    normalize_symbol,
    multi_model_exists,
    validate_stock,
    train_multi_model,
)


# =========================================================
# TRAINING JOB STORAGE
# =========================================================

training_jobs = {}

jobs_lock = threading.Lock()


# =========================================================
# CURRENT TIME
# =========================================================

def current_time():
    return datetime.now().isoformat(
        timespec="seconds"
    )


# =========================================================
# DEFAULT JOB
# =========================================================

def create_job_status(symbol):
    return {
        "symbol": symbol,
        "status": "idle",
        "stage": "Waiting",
        "message": "Model training has not started.",
        "started_at": None,
        "finished_at": None,
        "error": None,
    }


# =========================================================
# UPDATE JOB
# =========================================================

def update_job(
    symbol,
    **updates,
):
    with jobs_lock:

        if symbol not in training_jobs:

            training_jobs[symbol] = (
                create_job_status(
                    symbol
                )
            )

        training_jobs[
            symbol
        ].update(
            updates
        )


# =========================================================
# GET TRAINING STATUS
# =========================================================

def get_training_status(symbol):
    symbol = normalize_symbol(
        symbol
    )

    # -----------------------------------------------------
    # MODEL ALREADY EXISTS
    # -----------------------------------------------------

    if multi_model_exists(
        symbol
    ):

        return {
            "symbol": symbol,
            "status": "ready",
            "stage": "Ready",
            "message": (
                "Existing multi-horizon "
                "model is ready."
            ),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }


    # -----------------------------------------------------
    # EXISTING JOB
    # -----------------------------------------------------

    with jobs_lock:

        job = training_jobs.get(
            symbol
        )

        if job:

            return job.copy()


    # -----------------------------------------------------
    # NO JOB
    # -----------------------------------------------------

    return create_job_status(
        symbol
    )


# =========================================================
# BACKGROUND TRAINING WORKER
# =========================================================

def training_worker(symbol):
    try:

        # =================================================
        # VALIDATION
        # =================================================

        update_job(
            symbol,

            status="training",

            stage="Validating stock",

            message=(
                "Checking stock availability "
                "and historical market data."
            ),

            error=None,
        )


        validate_stock(
            symbol
        )


        # =================================================
        # DATA PREPARATION
        # =================================================

        update_job(
            symbol,

            status="training",

            stage="Preparing data",

            message=(
                "Preparing historical data "
                "and technical features."
            ),
        )


        # =================================================
        # TRAINING
        # =================================================

        update_job(
            symbol,

            status="training",

            stage="Training BiLSTM",

            message=(
                "Training the 1D, 3D, 5D "
                "and 10D Multi-Horizon BiLSTM."
            ),
        )


        train_multi_model(
            symbol
        )


        # =================================================
        # VERIFY MODEL
        # =================================================

        update_job(
            symbol,

            status="training",

            stage="Evaluating model",

            message=(
                "Checking trained model "
                "and evaluation files."
            ),
        )


        if not multi_model_exists(
            symbol
        ):

            raise RuntimeError(
                "Training finished but required "
                "model files were not found."
            )


        # =================================================
        # READY
        # =================================================

        update_job(
            symbol,

            status="ready",

            stage="Ready",

            message=(
                "Multi-horizon model trained, "
                "evaluated and saved successfully."
            ),

            finished_at=current_time(),

            error=None,
        )


    except Exception as e:

        update_job(
            symbol,

            status="error",

            stage="Training failed",

            message=(
                "Unable to prepare the "
                "multi-horizon forecast model."
            ),

            finished_at=current_time(),

            error=str(e),
        )


# =========================================================
# START TRAINING
# =========================================================

def start_training(symbol):
    symbol = normalize_symbol(
        symbol
    )


    # =====================================================
    # MODEL ALREADY EXISTS
    # =====================================================

    if multi_model_exists(
        symbol
    ):

        return {
            "symbol": symbol,
            "status": "ready",
            "stage": "Ready",
            "message": (
                "Existing multi-horizon "
                "model is already available."
            ),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }


    # =====================================================
    # CHECK FOR EXISTING ACTIVE JOB
    # =====================================================

    with jobs_lock:

        existing_job = (
            training_jobs.get(
                symbol
            )
        )


        if (
            existing_job
            and existing_job.get(
                "status"
            )
            ==
            "training"
        ):

            return existing_job.copy()


        # =================================================
        # CREATE NEW JOB
        # =================================================

        training_jobs[
            symbol
        ] = {
            "symbol": symbol,
            "status": "training",
            "stage": "Starting",
            "message": (
                "Preparing background "
                "model training."
            ),
            "started_at": current_time(),
            "finished_at": None,
            "error": None,
        }


    # =====================================================
    # START BACKGROUND THREAD
    # =====================================================

    thread = threading.Thread(
        target=training_worker,

        args=(
            symbol,
        ),

        daemon=True,
    )


    thread.start()


    return get_training_status(
        symbol
    )