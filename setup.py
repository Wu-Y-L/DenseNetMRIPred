
#installing dependencies for package

try:
    from pathlib import Path
except ModuleNotFoundError:
    print(f"pathlib not found, installing pathlib")
    %pip install -q pathlib

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    print(f"print torch not found, installing torch 2.8 cuda 12.9")
    %pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    print("Scikit learn not found, installing scikit learn")
    %pip install -q scikit-learn

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print("tqdm not found installing tqdm")
    %pip install -q tqdm

try:
    from torchmetrics import ConfusionMatrix
except ModuleNotFoundError:
    print("torchmetrics not found installing torchmetrics")
    %pip install -q torchmetrics

try:
    from mlxtend.plotting import plot_confusion_matrix
except ModuleNotFoundError:
    print("mlxtend not found, installing mlxtend")
    %pip install -q mlxtend

try:
    import zipfile 
except ModuleNotFoundError:
    print("install modulenotfound")
    %pip install -q zipfile

import zipfile 
package = Path(r"DenseNetMRIPred")
model_path = package / "best_model"

with zipfile.ZipFile("best_model.zip" ,'r') as zip_ref:
    zip_ref.extractall(model_path)

os.remove("best_model.zip")
