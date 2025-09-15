from fastapi import FastAPI
try:
    from backend.app.api.endpoints import train, predict, annotate, auto_annotation, projects, models, cloud_training
except ImportError:
    from app.api.endpoints import train, predict, annotate, auto_annotation, projects, models, cloud_training

app = FastAPI(title="YOLOv8 Service")

app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(train.router, prefix="/api/training", tags=["Training"])
app.include_router(cloud_training.router, prefix="/api/cloud-training", tags=["Cloud Training"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(annotate.router, tags=["Annotation"])
app.include_router(auto_annotation.router, prefix="/api/auto-annotation", tags=["Auto-Annotation"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 API"}