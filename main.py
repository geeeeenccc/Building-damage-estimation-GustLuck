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


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def extract_coordinates(results):
    coordinates = []
    for pred in results.pred:
        for det in pred:
            class_id = int(det[-1])
            confidence = det[4]
            bbox = det[:4].tolist()
            coordinates.append((class_id, confidence, bbox))
    return coordinates


@app.post("/upload_photo_of_building/")
async def upload_photo_of_building(request: Request, file: UploadFile = None):
    try:
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_path = file.filename

        model = torch.hub.load('yolov5', 'custom', 'yolov5/runs/train/exp8/weights/best.pt', source='local', force_reload=True)
        results = model(img_path)
        results.show()

        coordinates = extract_coordinates(results)

        total_area_1 = 0
        total_area_4 = 0
        total_area_3 = 0
        area_0 = 0
        for class_id, confidence, bbox in coordinates:
            if class_id == 0:
                area_0 = bbox_area(bbox) * 0.9
            elif class_id == 1:
                total_area_1 += bbox_area(bbox)
            elif class_id == 3:
                total_area_3 += bbox_area(bbox)
            elif class_id == 4:
                total_area_4 += bbox_area(bbox)

        result = 0
        if area_0 == 0:
            result = "The building is total crashed or there is no building at the picture."
        else:
            result = (total_area_1 + total_area_4 + total_area_3) / area_0 * 100
            if result >= 100:
                result = "There are multiple damaged objects or the number of damage is very big."

        os.remove(img_path)

        return templates.TemplateResponse("results.html", {"request": request, "result": result})
    except FileNotFoundError:
        return templates.TemplateResponse("index_error.html", {"request": request, "error": "You didn't upload the photo."})
