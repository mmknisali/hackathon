from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from model import SmartStarTrackerDenoiser
import os

app = FastAPI(title="Star Tracker Sensor Error Correction API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SmartStarTrackerDenoiser(
    input_size=7,
    hidden_size=128,
    num_layers=2,
    num_heads=4
).to(device)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'star_tracker_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
else:
    print("Warning: Model not found, using untrained model")

class SensorData(BaseModel):
    sequence: list

@app.get("/")
def root():
    return {"message": "Star Tracker Sensor Error Correction API", "status": "running"}

@app.post("/predict")
def predict(data: SensorData):
    try:
        sequence = np.array(data.sequence, dtype=np.float32)
        
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1, 7)
        elif sequence.ndim == 2:
            sequence = sequence.reshape(1, sequence.shape[0], 7)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequence).to(device)
            output = model(input_tensor)
        
        result = output.cpu().numpy().tolist()
        
        return {
            "input": data.sequence,
            "corrected": result[0] if isinstance(result[0], list) else result,
            "error_reduction": "computed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(sequences: list):
    try:
        results = []
        
        for seq in sequences:
            seq_array = np.array(seq, dtype=np.float32)
            if seq_array.ndim == 1:
                seq_array = seq_array.reshape(1, -1, 7)
            elif seq_array.ndim == 2:
                seq_array = seq_array.reshape(1, seq_array.shape[0], 7)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(seq_array).to(device)
                output = model(input_tensor)
            
            results.append(output.cpu().numpy().tolist()[0])
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": os.path.exists(model_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
