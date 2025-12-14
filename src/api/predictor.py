import time

def predict_text(app , text:str):
    if text.strip() == "":
        pred = -1
    # import pdb
    # pdb.set_trace()
    text = text.strip()
    text = text.lower()
    start_time_embedding = time.perf_counter()
    a = app.state.embed_model.encode([text], convert_to_numpy=True)
    end_time_embedding = time.perf_counter()
    total_time_embedding = end_time_embedding - start_time_embedding

    start_time_modelling = time.perf_counter()
    pred = app.state.clf.predict(a)
    end_time_modelling = time.perf_counter()
    total_time_modelling = end_time_modelling - start_time_modelling

    if pred == 0:
        classified = "Negative"
    elif pred == 1:
        classified = "Positive"
    else:
        classified = "Please re-right your movie review"
    
    predictions = {
        "type": "classification",
        "label": int(pred) ,
        "embedding_time" : float(total_time_embedding), 
        "prediction_time" : float(total_time_modelling),
        "model_version" : str(app.state.model_version)
    }

    return predictions