
import torch
from torch import nn
import torch.utils.data
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model : torch.nn.Module,
               train_dataloader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim,
               device : torch.device):
    """_summary_

    trains the model for one step

    Args:
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (torch.optim): _description_
        device (torch.device): _description_

    train_step takes in 5 parameters, the model, dataloader, a loss function, an optimizer, and the device for the training to happen on.
    in the function, a forward pass happens. EXAMPLE: y_pred = model(X) where X is a tensor and y_pred is the predicted value of the model
    then the prediction is passed through a loss function and optimizer is then zero.grad()'d and then backpropagation and gradient descent
    is performed
    The function returns 2 values, the train loss and train accuracy for experiment tracking
    """

    # initiate train loss and train accuracy
    total_loss, total_acc = 0, 0

    #run in model in training mode
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        #forwad pass
        y_pred = model(X)

        #calculate loss
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        total_acc += torch.eq(y_pred.argmax(dim=1), y).sum().item() / len(y)

        #optimizer zero grad
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # gradient descent
        optimizer.step()

    train_loss = total_loss / len(train_dataloader)
    train_acc = total_acc / len(train_dataloader) * 100

    return train_loss, train_acc

def test_step(model : nn.Module,
              test_dataloader : torch.utils.data.DataLoader,
              loss_fn : nn.Module,
              device : torch.device):
    """_summary_

    Args:
        model (nn.Module): model
        test_dataloader (torch.utils.data.DataLoader): dataloader
        loss_fn (nn.Module): loss function
        device (torch.device): device

    Returns:
        2 values

    similar to train step, calculates and returns a test loss and test acc
    """
    total_loss, total_acc = 0, 0

    # run model in eval and torch.inference_mode()
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y  = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += torch.eq(y_pred.argmax(dim=1), y).sum().item() / len(y)

    test_loss = total_loss / len(test_dataloader)
    test_acc = total_acc / len(test_dataloader) * 100

    return test_loss, test_acc

def train(model : nn.Module,
          train_dataloader : torch.utils.data.DataLoader,
          test_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim,
          scheduler : torch.optim.lr_scheduler,
          loss_fn : nn.Module,
          num_epochs : int = 20,
          device : torch.device = device):

    """_summary_
    num_epochs : the number of times train step and test step is run for
    combines previous two functions
    returns a dictionary of train_loss, train_acc, test_loss, test_acc
    """

    # initialize dictionary
    results = {"train_loss" : [],
               "train_acc" : [],
               "test_loss" : [],
               "test_acc" : []}

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(model = model, train_dataloader= train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer, device = device)
        test_loss, test_acc = test_step(model = model, test_dataloader = test_dataloader, loss_fn = loss_fn,
                                        device = device)

        #uses train loss as metric to reduce learning rate
        scheduler.step(train_loss)

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

        # print out epoch, train loss and acc, test loss and acc for experiment tracking

        print(f"epoch : {epoch}",
              "-"*30,
              f"\ntrain_loss : {train_loss:.4f} | train_acc : {train_acc:.2f}",
              f"\ntest_loss : {test_loss:.4f} | test_acc : {test_acc:.2f}")

    return results


