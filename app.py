
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"


# Load latest registered model from MLflow
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Latest registered model loaded successfully.")
        return model

    except Exception as e:
        print(f"Model loading failed: {e}")
        return None


model = load_latest_model()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
