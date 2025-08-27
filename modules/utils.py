### plots confusion matrix, 9 sets of prediction against truth label, loss and accuracy curve and save them to folder named results

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import torchvision
import torch
from typing import Dict
import matplotlib.pyplot as plt
from pathlib import Path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_loss_curves(results: Dict, epochs : int , curve_name : str):
    """_summary_

    produces loss curves

    Args:
        results (Dict): dictionary consisting of train_loss, train_acc, test_loss, test_acc
        epochs (int) : number of epochs the training loop ran for
        curve_name (str) : the name of the curve to be named

    uses a results dictionary consisting of 4 variables to produce 2 graphcs, a loss curve and an accuracy curve
    """

    num_epochs = [epoch for epoch in range(epochs)]

    plt.figure(figsize = (10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(num_epochs, results["train_loss"], color = "blue", label = "train loss")
    plt.plot(num_epochs,results["test_loss"], color = "orange", label = "test loss")
    plt.title("loss curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(num_epochs, results["train_acc"], color = "blue", label = "train accuracy")
    plt.plot(num_epochs, results["test_acc"], color = "orange", label = "test_accuracy")
    plt.title("accuracy curve")
    plt.legend()

    plt.savefig(fname = curve_name, format = "png")

def save_model(model : torch.nn.Module,
               target_dir : str,
               model_name : str):

  MODEL_PATH = Path(target_dir)
  MODEL_PATH.mkdir(parents = True, exist_ok = True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  torch.save(obj = model.state_dict(), f = MODEL_PATH / model_name)


