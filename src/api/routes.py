from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import numpy as np
import logging
from datetime import datetime

from ..ml.trainer import ModelTrainer
from ..ml.inference import InferenceEngine
from ..storage.model_store import ModelStore
from ..monitoring.metrics import MetricsCollector
from ..data.loader import DataLoader

logger = logging.getLogger(__name__)

router = APIRouter()
model_store = ModelStore()
metrics_collector = MetricsCollector()


# Pydantic models for request validation
class TrainingRequest(BaseModel):
    model_type: str = Field(..., description="Type of model: 'linear' or 'neural_net'")
    input_dim: int = Field(..., gt=0, description="Input dimension")
    output_dim: int = Field(1, gt=0, description="Output dimension")
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=1024)
    learning_rate: float = Field(0.01, gt=0, lt=1)
    hidden_dims: Optional[List[int]] = Field(
        None, description="Hidden layer dimensions for neural_net"
    )

    @validator("model_type")
    def validate_model_type(cls, v):
        if v not in ["linear", "neural_net"]:
            raise ValueError('model_type must be "linear" or "neural_net"')
        return v


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    data: List[List[float]] = Field(..., description="Input data as 2D array")

    @validator("data")
    def validate_data(cls, v):
        if not v or not v[0]:
            raise ValueError("data cannot be empty")
        return v


class ModelResponse(BaseModel):
    model_id: str
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    model_id: str
    predictions: List[float]
    confidence: Optional[List[float]] = None
    inference_time_ms: float


# Health check
@router.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": model_store.count_models(),
    }


# Training endpoints
@router.post("/train", response_model=ModelResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model"""
    try:
        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting training for {model_id}")
        metrics_collector.record_event("training_started", {"model_id": model_id})

        # Create synthetic training data for demonstration
        # In production, this would load from uploaded data
        X_train = np.random.randn(1000, request.input_dim)
        y_train = np.random.randn(1000, request.output_dim)
        X_val = np.random.randn(200, request.input_dim)
        y_val = np.random.randn(200, request.output_dim)

        # Initialize and train
        config = request.dict()
        trainer = ModelTrainer(config)
        result = trainer.train(X_train, y_train, X_val, y_val)

        # Save model
        metadata = {
            "config": config,
            "training_result": result,
            "created_at": datetime.utcnow().isoformat(),
        }
        model_store.save_model(model_id, trainer.model, metadata)

        metrics_collector.record_event(
            "training_completed",
            {"model_id": model_id, "final_loss": result["final_loss"]},
        )

        return ModelResponse(
            model_id=model_id,
            status="success",
            message="Model trained successfully",
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        metrics_collector.record_event("training_failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/train/upload")
async def train_from_upload(file: UploadFile = File(...), config: str = None):
    """Train model from uploaded CSV file"""
    try:
        # Save uploaded file temporarily
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load data
        loader = DataLoader()
        X, y = loader.load_from_csv(tmp_path, target_column="target")

        # Split data
        splits = loader.split_data(X, y)

        # Parse config
        training_config = json.loads(config) if config else {}
        training_config["input_dim"] = X.shape[1]
        training_config["output_dim"] = 1

        # Train model
        trainer = ModelTrainer(training_config)
        result = trainer.train(
            splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]
        )

        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        model_store.save_model(
            model_id, trainer.model, {"config": training_config, "result": result}
        )

        return {"model_id": model_id, "status": "success", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoints
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        start_time = datetime.utcnow()

        # Load model
        model, metadata = model_store.load_model(request.model_id)
        if model is None:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model_id} not found"
            )

        # Prepare data
        X = np.array(request.data)

        # Inference
        engine = InferenceEngine(model)
        result = engine.predict_with_confidence(X)

        # Calculate inference time
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        metrics_collector.record_metric("inference_time_ms", inference_time)
        metrics_collector.record_event(
            "prediction_made", {"model_id": request.model_id, "num_samples": len(X)}
        )

        return PredictionResponse(
            model_id=request.model_id,
            predictions=result["predictions"].flatten().tolist(),
            confidence=result["confidence"].tolist(),
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(model_id: str, file: UploadFile = File(...)):
    """Batch predictions from CSV file"""
    try:
        import tempfile
        import pandas as pd

        # Save and load file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        X = df.values

        # Load model and predict
        model, _ = model_store.load_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        engine = InferenceEngine(model)
        predictions = engine.predict_batch(X)

        return {
            "model_id": model_id,
            "num_predictions": len(predictions),
            "predictions": predictions.flatten().tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@router.get("/models")
async def list_models():
    """List all available models"""
    try:
        models = model_store.list_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    try:
        _, metadata = model_store.load_model(model_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {"model_id": model_id, "metadata": metadata}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    try:
        success = model_store.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        metrics_collector.record_event("model_deleted", {"model_id": model_id})
        return {"message": f"Model {model_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoints
@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = metrics_collector.get_all_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{metric_name}")
async def get_specific_metric(metric_name: str):
    """Get a specific metric"""
    try:
        metric = metrics_collector.get_metric(metric_name)
        if metric is None:
            raise HTTPException(
                status_code=404, detail=f"Metric {metric_name} not found"
            )
        return {metric_name: metric}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
