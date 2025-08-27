
import torch 
import matplotlib.pyplot as plt 
import torchvision 
from modules import model_builder, datasetup
from torchvision import transforms
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description= "for adjusting figure size and font size")
parser.add_argument("--figure_size", help="figure size please input an integer")
parser.add_argument("--font_size", help="font size, please input an integer")

args = parser.parse_args()
figure_size = args.figure_size
font_size = args.font_size

figure_size = int(figure_size)
font_size = int(font_size)

package = Path("DenseNetMRIPred")
best_model = package / "best_model"
data_path = package / "data"
for_pred = package / "for_pred"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for _, _, filename in os.walk(best_model):
    file = filename

file_name_list = []
for _, _, filename in os.walk(for_pred):
    file_name_list.extend(filename)

default_transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(dataset_dir = data_path, transforms = default_transform, BATCH_SIZE = 32)
    
prediction_model = model_builder.initiate_dense_model(class_names, device = device)

file_path = best_model / file[0]

prediction_model.load_state_dict(torch.load(file_path, weights_only = True) if torch.cuda.is_available() else torch.load(file_path, map_location = torch.device('cpu'), weights_only = True))
prediction_model.eval()

#get data from for predictions folder 
prediction_data = torchvision.datasets.ImageFolder(root = for_pred, transform = default_transform)
prediction_dataloader = torch.utils.data.DataLoader(prediction_data, shuffle = False, batch_size = 1)

y_preds = []
with torch.inference_mode():
    for X, y in prediction_dataloader:
      X, y = X.to(device), y.to(device)
      y_pred = prediction_model(X)
      y_pred_label = y_pred.argmax(dim = 1)
      y_preds.append(y_pred_label.item() if isinstance(y_pred_label, torch.Tensor) else y_pred_label)

plt.figure(figsize = (figure_size, figure_size))

x = round(len(file_name_list) / 3) if len(file_name_list) % 3 >= 2 else round(len(file_name_list) / 3 ) + 1

for i in range(len(y_preds)):
    plt.subplot(x, 3, i + 1)
    plt.imshow(prediction_data[i][0].permute(1, 2, 0))
    plt.title(class_names[y_preds[i]], fontsize = font_size)
    plt.xlabel(f"file name : {file_name_list[i]}", fontsize = font_size)
    plt.xticks([])
    plt.yticks([])

save_path = for_pred / "predictions.png"
plt.savefig(fname = save_path, format = "png")
