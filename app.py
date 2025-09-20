from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import uvicorn
import io

# Load YOLO model (relative path so it works in Azure too)
model = YOLO("runs/detect/train/weights/best.pt")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "YOLO model is running on Azure!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(io.BytesIO(await file.read()))

    # Run YOLO inference
    results = model.predict(image)

    # Collect detections
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()
        })

    return {"detections": detections}

if __name__ == "__main__":
    # Important for Azure: bind to 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000)
