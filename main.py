import torch
import shutil
import os
import uvicorn
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates


app = FastAPI(
    title="Calculate Building's Damage"
)
templates = Jinja2Templates(directory='templates')


@app.post("/upload_photo_of_building/")
async def upload_photo_of_building(request: Request, file: UploadFile):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img_path = file.filename

    model = torch.hub.load('yolov5', 'custom', 'yolov5/runs/train/exp8/weights/best.pt', source='local', force_reload=True)
    results = model(img_path)
    results.show()
    os.remove(img_path)

    return templates.TemplateResponse("results.html", {"request": request})
