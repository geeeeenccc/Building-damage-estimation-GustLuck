# Building damage estimation by team "Gust Luck"

This FastAPI application utilizes YOLOv5 object detection to assess the extent of damage to buildings based on uploaded images. It provides a percentage estimation of the damage along with descriptive information about the detected damages.

## Features

**Damage Assessment:** The application detects various types of damages such as regular damage, damage to the roof, and broken windows in uploaded images of buildings.

**Severity Estimation:** It calculates the severity of the damage based on the detected areas and provides a percentage of damage.

**Descriptive Insights:** Provides descriptive information about the detected damages and suggests possible conclusions based on the severity.

## Installation

**1. Clone the repository to your folder:**
```commandline
git clone https://github.com/geeeeenccc/Building-damage-estimation-GustLuck.git .
```

**2. Install dependencies:**

```commandline
cd yolov5
```

```commandline
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```commandline
uvicorn main:app --reload
```

2. Run **index.html** file.

3. Upload an image of a building for damage assessment.

4. View the calculated damage percentage, descriptive information, and conclusions.

## Configuration

**Model:** The application uses a YOLOv5 model for object detection. You can replace the model path in the **upload_photo_of_building** function in **main.py** with your custom YOLOv5 model path.

![image](https://github.com/geeeeenccc/Building-damage-estimation-GustLuck/assets/101811004/6b2ba63b-a1f1-4dc3-8316-58fdfed4aa08)


![image](https://github.com/geeeeenccc/Building-damage-estimation-GustLuck/assets/101811004/9377f82b-c9f1-4cd7-a9c2-990ab7704f02)


![image](https://github.com/geeeeenccc/Building-damage-estimation-GustLuck/assets/101811004/54c6842d-1b11-48d9-a9a3-e344f1fb6d00)

