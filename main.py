import torch
import shutil
import os
import uvicorn
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from collections import Counter


app = FastAPI(
    title="Calculate Building's Damage"
)
templates = Jinja2Templates(directory='templates')


def bbox_area(bbox):
    # Calculate the area of a bounding box
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def extract_coordinates_and_counts(results):
    # Extract coordinates and count occurrences of each class
    coordinates = []
    class_counts = Counter()
    for pred in results.pred:
        for det in pred:
            class_id = int(det[-1])
            confidence = det[4]
            bbox = det[:4].tolist()
            coordinates.append((class_id, confidence, bbox))
            class_counts[class_id] += 1
    return coordinates, class_counts


@app.post("/upload_photo_of_building/")
async def upload_photo_of_building(request: Request, file: UploadFile = None):
    try:
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_path = file.filename

        # Load model from local weights
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/best.pt', force_reload=True)
        results = model(img_path)
        results.show()

        coordinates, class_counts = extract_coordinates_and_counts(results)

        total_area_1 = 0  # area of regular damage
        total_area_3 = 0  # area of damage to the roof
        total_area_4 = 0  # area of broken window
        area_0 = 0  # total area of the building

        for class_id, confidence, bbox in coordinates:
            if class_id == 0:
                area_0 += bbox_area(bbox) * 0.85
            elif class_id == 1:
                total_area_1 += bbox_area(bbox)
            elif class_id == 3:
                total_area_3 += bbox_area(bbox)
            elif class_id == 4:
                total_area_4 += bbox_area(bbox)

        result = 0  # percentage of damage
        if area_0 == 0:
            result_info = "The building is totally crashed or there is no building in the picture."
        else:
            result = (total_area_1 + total_area_4 + total_area_3) / area_0 * 100
            if result < 0:
                result = 0
            if result >= 100:
                result_info = "There are multiple damaged objects or the number of damage is very big. "
            elif result > 30:
                result_info = "Major damage to the building. "
            elif result > 15:
                result_info = "Noticeable damage to the building. "
            elif result > 0:
                result_info = "Minimal damage to the building. "
            else:
                result_info = "The building appears in good condition. "

            # Insert window status here
            if class_counts[4] == 0:
                result_info += "No broken windows were detected. "
            elif 0 < class_counts[4] <= 10:
                result_info += "Few broken windows detected (less than 10). "
            elif class_counts[4] > 10:
                result_info += "Many broken windows detected (more than 10). "

            # Add concluding advice based on damage severity
            if result >= 100 or result > 30:
                result_info += "\nUrgent professional assessment and repairs needed."
            elif result > 15:
                result_info += "\nEvaluation recommended to determine necessary repairs."
            elif result > 0:
                result_info += "\nSimple repairs likely sufficient; inspection advised."
            else:
                result_info += "\nRegular maintenance recommended."

        os.remove(img_path)

        return templates.TemplateResponse("results.html", {"request": request, "result": round(result, 4), "result_info": result_info})
    except FileNotFoundError:
        return templates.TemplateResponse("index_error.html", {"request": request, "error": "You didn't upload the photo."})


if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port=8000, reload=True)
