import os
import torch
from torch import nn
from torch.onnx import export
import onnxruntime as ort
from matplotlib import pyplot as plt

def measure_time(start, end, device=None, stepper=False):
    measurement = f"{(end-start):.2f} sec"
    if stepper:
        print(f'[*] Step Run Time : {measurement}')
    else:
        print(f'''
----- Time -----
[*] Device : {device}
[*] Total Run Time : {measurement}
----------------
''')


def measure_accuracy(y_logit, y_true):
    pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
    acc = (pred_class == y_true).sum().item()/len(y_logit)
    return acc

def PerformanceGraph(results):
    loss = results['train_loss']
    test_loss = results['val_loss']
    acc = results['train_acc']
    test_acc = results['val_acc']
    epochs = results['epoch']

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, test_loss, label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Train acc')
    plt.plot(epochs, test_acc, label='Val acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.close


def save_model(model: nn.Module, 
               model_name: str, 
               format:str,
               device:str ='cuda' if torch.cuda.is_available() else "cpu",
               **kwargs):
    
    abs_path = os.path.join(os.getcwd(), "trained_models")
    os.makedirs(abs_path, exist_ok=True)
    model_path = os.path.join(abs_path, model_name)
    if format.lower() == '.pt':
        torch.save(
            obj=model.state_dict(),
            f=model_path
        )
        print(f"[*] PT Model successfully saved to {model_path}")
        return model_path
    
    elif format.lower() == '.onnx':
        dummy_input = kwargs['dummy_input']
        model.eval()
        model.to(device)
        export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=18,
            input_names=['input'],
            output_names=['output'],
            dynamic_shapes={
                'x': {0: 'batch_size'},   # Specifies axis 0 as dynamic 'batch_size'
            }
        )
        print(f"[*] ONNX model successfully exported to {model_path}")
        return model_path
    else:
        raise ValueError(f"Unsupported format '{format}'. Please use '.pt' or '.onnx'.")



def load_model(model_path: str, 
               format:str,
               device:str='cuda' if torch.cuda.is_available() else "cpu", 
               **kwargs):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if format.lower() == '.pt':
        input_shape = kwargs['input_shape']
        output_shape = kwargs['output_shape']
        hidden_units = kwargs['hidden_units']
        model_class = kwargs['model_class']

        if model_class is None:
            raise ValueError("You must provide the `model_class` argument to load a PyTorch (.pt) state dict.")
        model:nn.Module = model_class(input_shape=input_shape, 
                            output_shape=output_shape, 
                            hidden_units=hidden_units).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"[*] PT Model successfully loaded from {model_path}")
        return model

    elif format.lower() == '.onnx':
        providers = None
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(model_path, providers=providers)
        print(f"[*] ONNX Model successfully loaded from {model_path}")
        return ort_session
    else:
        raise ValueError(f"Unsupported format '{format}'. Please use '.pt' or '.onnx'.")

if __name__ == '__main__':
    from torch import nn
    r = torch.rand(1,3,224,224)
    hidden_units = 5

    class CustomModel(nn.Module):
        def __init__(self, input_shape, output_shape, hidden_units: int = 16):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=2)
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(in_features=hidden_units * 223 * 223, out_features=output_shape)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.flatten(x)
            x = self.linear(x)
            return x

    model = CustomModel
    # print(model(r).shape)
    # save_model(model=model, model_name='dummy_model.pt', format='.pt')
    load_model(model_class=model, 
               model_path=r"D:\STNN_Novel Synapse Transformer\pytorch Turorial Udemy\06_Pytorch_Going_Modular\trained_models\dummy_model.pt",
               format='.pt',
               device='cpu',
               input_shape=3,
               output_shape=1,
               hidden_units=hidden_units
               )