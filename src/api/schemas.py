from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    embed_model_loaded: bool
    model_version: str| None 

