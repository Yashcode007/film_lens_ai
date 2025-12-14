from predictor import predict_text
from fastapi import FastAPI
from app_state  import init_app_state
from schemas import (HealthResponse)
app = FastAPI()


@app.on_event("startup")
def startup_event():
    init_app_state(app)


@app.post("/predict")
def prediction_model(review:str):
    result = predict_text(app , review)
    return result 

# @app.post("/health")
# def health_check():
#     is_clf_loaded = hasattr(app.state, "clf")
#     is_embed_loaded = hasattr(app.state, "embed_model")

#     return HealthResponse(
#         "status" = "ok" if (is_clf_loaded and is_embed_loaded) else "error" , 
#         "classifier_loaded" =  is_clf_loaded, 
#         "embed_model_loaded" =  is_embed_loaded,
#         "model_version" = getattr(app.state, "model_version", None)
#     )
