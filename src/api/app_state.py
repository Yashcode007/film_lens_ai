### Load and Expose model artifacts and data on app startup(embedding model reference , classifier , label map , train embeddings)

## Mental Model:-  In notebook world , the variables live in memory inside jupyter cells and everything is interactive 
## In API world , the variables must live in app.state , they load once and all reuqests reuse them 
# When FastAPI starts, it runs the startup event.
# During that startup event, you call init_app_state(app).

from sentence_transformers import SentenceTransformer
import joblib
import numpy as np 
import pandas as pd


def load_classifier(path:str) :
    loaded_model = joblib.load(path)
    return loaded_model

def get_embedding_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

def init_app_state(app):
    clf = load_classifier(r"C:\Users\Yashita\Desktop\film_lens_ai\models\cls_logistic_v1.joblib")
    embed_model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

    app.state.clf = clf
    app.state.embed_model = embed_model
    app.state.model_version = "v1"
    

