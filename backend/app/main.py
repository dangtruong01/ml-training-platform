from fastapi import FastAPI
from backend.app.api.endpoints import train, predict, annotate, auto_annotation

app = FastAPI(title="YOLOv8 Service")

app.include_router(train.router, tags=["Training"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(annotate.router, tags=["Annotation"])
app.include_router(auto_annotation.router, prefix="/api/auto-annotation", tags=["Auto-Annotation"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 API"}