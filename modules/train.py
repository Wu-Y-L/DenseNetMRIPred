
import torch
import torchvision
import model_builder, utils, datasetup, engine
import argparse
import torchvision
from torchvision import transforms
from pathlib import Path
from torch import nn
from tqdm import tqdm

parser = argparse.ArgumentParser(description = "for training arguments")
parser.add_argument("--num_epochs", type = int,help = " the number of epochs to train for")
parser.add_argument("--batch_size", type =int , help = " the batch size of dataloaders")
parser.add_argument("--learning_rate", type = float, help = " the learning rate of the optimizer")
parser.add_argument("--curve_name" , type = str, help = "the name of the loss curve produced")
parser.add_argument("--model_name", type = str, help = "the name of the model saved")

args = parser.parse_args()
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
curve_name = args.curve_name
model_name = args.model_name
device = 'cuda' if torch.cuda.is_available() else 'cpu'

package = Path(r"DenseNetMRIPred")
modules = package / "modules"
data_path = package / "data"
graph_path = package / "graphs"

default_transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(dataset_dir = data_path, transforms = default_transform, BATCH_SIZE = batch_size)

model = model_builder.initiate_dense_model(class_names, device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = 1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.8, patience = 10)

results = engine.train(model = model,
                       train_dataloader = train_dataloader,
                       test_dataloader = test_dataloader,
                       loss_fn = loss_fn,
                       optimizer = optimizer,
                       scheduler = scheduler,
                       device = device,
                       num_epochs = num_epochs)

curve_path = graph_path / curve_name

utils.plot_loss_curves(results = results, epochs = num_epochs, curve_name = curve_path)
utils.save_model(model = model, target_dir = "saved_model", model_name = model_name)
