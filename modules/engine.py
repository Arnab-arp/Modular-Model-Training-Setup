import torch
from modules.utility import measure_accuracy

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device):

    train_loss, train_accuracy = 0
    model.train()

    for batch, (X,y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        y_logit = model(X)
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        # train_accuracy += (pred_class == y).sum().item()/len(y_logit)
        train_accuracy += measure_accuracy(y_logit=y_logit, y_true=y)

    train_loss = train_loss/len(data_loader)
    train_accuracy = train_accuracy/len(data_loader)

    return train_loss, train_accuracy



def eval_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,  
               device: torch.device):
    val_loss, val_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_logit_val = model(X)
            loss = loss_fn(y_logit_val, y)
            val_loss += loss.item()

            val_accuracy += measure_accuracy(y_logit=y_logit_val, y_true=y)
    val_loss = val_loss/len(data_loader)
    val_accuracy = val_accuracy/len(data_loader)

    return val_loss, val_accuracy

def EvaluateOnTest(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,  
               device: torch.device):
    test_accuracy = 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_logit_val = model(X)

            test_accuracy += measure_accuracy(y_logit=y_logit_val, y_true=y)
    test_accuracy = test_accuracy/len(data_loader)

    return test_accuracy
        