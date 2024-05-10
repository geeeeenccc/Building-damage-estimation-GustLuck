import torch
import shutil
import os
from fastapi import FastAPI, UploadFile


app = FastAPI(
    title="Calculate Building's Damage"
)


@app.post("/upload_photo_of_building/")
def upload_photo_of_building(file: UploadFile):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img_path = file.filename

    model = torch.hub.load('yolov5', 'custom', 'runs/train/exp/weights/yolov5s.pt', source='local', force_reload=True)
    results = model(img_path)
    results.show()

    os.remove(img_path)

    return {"status": "Photo was successfully analyzed."}
