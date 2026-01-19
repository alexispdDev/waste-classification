import uvicorn
from fastapi import FastAPI
from waste_classification import predict


app = FastAPI(title='waste-classification')


@app.get("/")
def root():
    return {"message": "Waste Classification Service"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=predict.PredictResponse)
def predict_endpoint(request: predict.PredictRequest):
    predictions, top_class, top_prob = predict.predict_waste(str(request.url))
    
    return predict.PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)