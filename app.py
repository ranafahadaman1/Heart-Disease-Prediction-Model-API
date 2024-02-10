from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["192.168.1.14"],  # Update this to the specific IP address of your iPhone
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the trained model
model1 = load('./models/linear_regression_model.sav')
model2 = load('./models/random_forest_classifier_model.sav')

class InputData(BaseModel):
    Age: int
    Sex: int
    RestingBP: int
    ChestPainType: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int  
    Oldpeak: float
    ST_Slope: int   

@app.get('/')
def home():
    return {'welcome':"Heart Disease Prediction API Model", 'models':"Make predictions using Linear Regression Model (/predictLR) or Random Forest (/predictRF)", 'bestModel':"Random Forest is performing the best."}


@app.post('/predictLR')
def predictLR(input_data: InputData):
    try:
        # Convert input data to NumPy array
        features = np.array([
            input_data.Age, input_data.Sex, input_data.ChestPainType, input_data.RestingBP,
            input_data.Cholesterol, input_data.FastingBS, input_data.RestingECG, input_data.MaxHR,
            input_data.ExerciseAngina, input_data.Oldpeak, input_data.ST_Slope
        ]).reshape(1, -1)

        # Make prediction
        prediction = model1.predict(features)

        # Determine prediction condition
        prediction_condition = "Person has heart disease!" if prediction[0] > 0.6 else "Person does not have heart disease!"

        # Return prediction and condition as JSON
        return {'prediction': prediction.tolist(), 'condition': prediction_condition}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predictRF')
def predictRF(input_data: InputData):
    try:
        # Convert input data to NumPy array
        features = np.array([
            input_data.Age, input_data.Sex, input_data.ChestPainType, input_data.RestingBP,
            input_data.Cholesterol, input_data.FastingBS, input_data.RestingECG, input_data.MaxHR,
            input_data.ExerciseAngina, input_data.Oldpeak, input_data.ST_Slope
        ]).reshape(1, -1)

        # Make prediction
        prediction = model2.predict(features)

         # Determine prediction condition
        prediction_condition = "Person has heart disease!" if prediction[0] > 0 else "Person does not have heart disease!"


        # Return prediction as JSON
        return {'prediction': prediction.tolist(), 'condition': prediction_condition}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add route to serve index.html
@app.get('/predictor', response_class=HTMLResponse)
async def get_predictor(request: Request):
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
