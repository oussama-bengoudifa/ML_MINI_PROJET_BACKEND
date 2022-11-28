# Importing necessary libraries
import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


# Initializing the fast API server
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading up the trained model
model = pickle.load(open('./model/model.pkl', 'rb'))

# Defining the model input types
class Candidate(BaseModel):
    cholesterol: int
    glucose: int
    hdl_chol: int
    chol_hdl_ratio: float
    age: int
    gender: float
    height: int
    weight: int
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    waist: int
    hip: int
    waist_hip_ratio: float

# Setting up the home route
@app.get("/")
async def root():
    return {"data": "TP DIF"}

@app.post("/prediction/")
async def get_predict(data: Candidate):
    sample = [[
        data.cholesterol,
        data.glucose,
        data.hdl_chol,
        data.chol_hdl_ratio,
        data.age,
        data.gender,
        data.height,
        data.weight,
        data.bmi,
        data.systolic_bp,
        data.diastolic_bp,
        data.waist,
        data.hip,
        data.waist_hip_ratio,
    ]]
    
    predicted_data = model.predict(np.array(sample))

    return {
        "data": {
            'prediction': predicted_data[0],
        }
    }

# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')