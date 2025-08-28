import os 
import zipfile
from pathlib import Path


model_path = Path("best_model")

with zipfile.ZipFile("best_model.zip" ,'r') as zip_ref:
    zip_ref.extractall(model_path)

os.remove("best_model.zip")
