import torch
import numpy as np
import onnxruntime as ort
from tqdm.auto import tqdm
from timeit import default_timer as timer
from modules.utility import measure_accuracy, measure_time

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device):

    train_loss, train_accuracy = 0, 0
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


def EvaluateOnTest_pt(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,  
               device: torch.device):
    accuracy = 0
    model.eval()
    s = timer()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader), desc='Evaluating PT Model'):
            X, y = X.to(device), y.to(device)
            y_logit_val = model(X)

            accuracy += measure_accuracy(y_logit=y_logit_val, y_true=y)
    mean_accuracy = accuracy/len(data_loader)
    measure_time(start=s, end=timer(), stepper=False, device=device)
    return mean_accuracy


def EvaluateOnTest_onnx(ort_session:ort.InferenceSession, 
                        data_loader: torch.utils.data.DataLoader):
    input_name = ort_session.get_inputs().name
    output_name = ort_session.get_outputs().name
    total_correct = 0
    s = timer()
    for X, y in tqdm(data_loader, desc='Evaluating ONNX Model'):
        x_numpy = X.detach().cpu().numpy().astype(np.float32)
        y_numpy = y.detach().cpu().numpy()

        outputs = ort_session.run([output_name], {input_name: x_numpy})
        y_logits = outputs[0]
        preds = np.argmax(y_logits, axis=1)
        total_correct += (preds == y_numpy).sum()/len(y_logits)
    mean_accuracy = total_correct / len(data_loader)
    measure_time(start=s, end=timer(), stepper=False, device='cpu')
    return mean_accuracy