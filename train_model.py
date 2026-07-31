from argparse import ArgumentParser
from modules.model_registry import Registry
import torch
import gc
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchinfo import summary
from modules.data_loaders import LoadDataFromPath, LoadDataSplit
from modules.engine import train_step, eval_step, EvaluateOnTest_pt, EvaluateOnTest_onnx
from modules.utility import  measure_time, PerformanceGraph, save_model, load_model

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int):
    
    history = {'epoch':[],
                'train_loss':[],
                'train_acc':[],
                'val_loss': [],
                'val_acc': []}
    b_start = timer()

    for epc in tqdm(range(1, epoch+1)):
        s_start = timer()
        train_loss, train_acc = train_step(model=model, optimizer=optimizer,
                                            loss_fn=loss_fn, data_loader=train_loader,
                                            device=device)
        val_loss, val_acc = eval_step(model=model, loss_fn=loss_fn,
                                        data_loader=val_loader, device=device)

        history['epoch'].append(epc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch : {epc}")
        print(f"Train Accuracy : {(train_acc*100):.2f}%  |  Train Loss : {train_loss:.5f}")
        print(f"Validation Accuracy : {(val_acc*100):.2f}%  |  Validation Loss : {val_loss:.5f}")

        measure_time(start=s_start, end=timer(), stepper=True)
    measure_time(start=b_start, end=timer(), stepper=False, device=device)
    return model, history


def main():
    Registry.discover_models('modules.models')
    registered_models = Registry.registered_models()

    parser = ArgumentParser(description="Simple Command-line tool for Model Training.\nType --help or -h for more commands")

    parser.add_argument("-m","--model", metavar='NAME', help=f"(Case Sensitive) Model to train. Models Listed : {registered_models.keys()}")
    parser.add_argument("-hu","--hidden-units", metavar='NAME', help="(Hyper Parameter) Initializes the model's hidden layers. *Some Models may have pre-initialized hidden units")
    parser.add_argument("-dp","--dir-path", metavar='PATH', help="(Optional) Sets the data path. *Only Use if data is not splitted")
    parser.add_argument("-dtr","--dir-train", metavar='PATH', help="(Optional) Path to train dataset")
    parser.add_argument("-dte","--dir-test", metavar='PATH', help="(Optional) Path to test dataset")
    parser.add_argument("-dv","--dir-val", metavar='PATH', help="(Optional) Path to validation dataset")
    parser.add_argument("-lr","--learning-rate", metavar='FLOAT', help="(Hyper Parameter) Sets the learning rate of the model")
    parser.add_argument("-bs","--batch-size", metavar='INT', help=f"(Hyper Parameter) Sets the batch size of the data")
    parser.add_argument("-epc","--epoch", metavar='INT', help="(Hyper Parameter) Train for N epochs")
    parser.add_argument("-v","--verbose", metavar='BOOL', help="Prints logs")
    parser.add_argument("-sa","--save-as", metavar='NAME', help="Name Of the Model")
    parser.add_argument("-f","--format", metavar='NAME', help="Sets the format of the model. [.pt, .onnx, ]")
    parser.add_argument("-d","--device", metavar='NAME', help="Set device ['cuda' or 'cpu']")

    model_name = None
    hidden_unit = None
    dir_path = None
    train_dir = None
    test_dir = None
    val_dir = None
    lr = None
    batch_size = None
    epoch = None
    verbose = True
    save_name = None
    format = None
    device = None


    args = parser.parse_args()
    parse_dict = dict(vars(args).items())
    for arg_name, val in vars(args).items():

        if arg_name == 'model':
            model_name = val

        if arg_name == 'hidden_units':
            hidden_unit = int(val)

        if arg_name == 'dir_path':
            dir_path = val
            parse_dict['dir_train'] = None
            parse_dict['dir_test'] = None
            parse_dict['dir_val'] = None

        if arg_name == 'dir_train':
            train_dir = val
            parse_dict['dir_path'] = None
            
        if arg_name == 'dir_test':
            test_dir = val
            parse_dict['dir_path'] = None

        if arg_name == 'dir_val':
            val_dir = val
            parse_dict['dir_path'] = None
        
        if arg_name == 'learning_rate':
            lr = float(val)

        if arg_name == 'batch_size':
            batch_size = int(val)

        if arg_name == 'epoch':
            epoch = int(val)

        if arg_name == 'verbose':
            verbose = True if val.lower() == 'true' else False

        if arg_name == 'save_as':
            save_name = val

        if arg_name == 'format':
            format = val if val in ['.pt', '.onnx'] else '.pt'

        if arg_name == 'device':
            device = val.lower()

    model_save_name = save_name + format

    if device == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA not found. Defaulting to CPU')
            device = 'cpu'

    # load data
    if dir_path is not None:
        data = LoadDataSplit(data_dir=dir_path, verbose=verbose, batch_size=batch_size)
        train_loader, val_loader, test_loader, input_shape, output_shape, classes, cls2idx = data
    elif dir_path is None and train_dir and test_dir and val_dir:
        data = LoadDataFromPath(data_dir=dir_path, verbose=verbose, batch_size=batch_size)
        train_loader, val_loader, test_loader, input_shape, output_shape, classes, cls2idx = data

    if not train_loader or not test_loader or not val_loader or not input_shape or not output_shape:
        raise ValueError("Data Did Not Load")
    
    # creating dummy inputs for torchinfo
    dummy_img, _ = next(iter(train_loader))
    dummy_input = torch.rand_like(dummy_img).to(device)

    # Initialize model
    model = registered_models[model_name](input_shape=input_shape, 
                                   output_shape=output_shape, 
                                   hidden_units=hidden_unit).to(device)
    # Initialize optimizer, loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    print(f"""
----- Description -----
Model : {model_name}
Save Name : {model_save_name}
Device : {device}
Loss Function : {loss_fn}
Optimizer : {optimizer}
-----------------------

----- Hyperparameters -----
Batch Size : {batch_size}
Epoch : {epoch}
Learning Rate : {lr}
Input Shape : {input_shape}
Output Shape : {output_shape}
Hidden Units: {hidden_unit}
---------------------------

----- Loaders -----
Classes : {classes}
Class Indexes : {cls2idx}
Image Shape : {dummy_input.shape}
Loaders -|> Train : {len(train_loader)}
         |> Validation : {len(val_loader)}
         |> Test : {len(test_loader)}
-------------------
[*] Performing Smoke Test
""")
    summary(model=model, input_data=dummy_input)
    print("[*] Smoke Test Passed\n[*] Starting Training")
    # train model
    model, history = train_model(model=model, 
                                train_loader=train_loader,
                                val_loader=val_loader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device,
                                epoch=epoch)

    # save model
    model_path = save_model(model=model, 
               model_name=model_save_name, 
               format=format,
               device=device,
               dummy_input=dummy_input)

    del model
    gc.collect()

    PerformanceGraph(results=history)

    model_class = registered_models[model_name]

    loaded_model = load_model(model_path=model_path, 
                format=format,
                device=device,
                input_shape=input_shape,
                output_shape=output_shape,
                hidden_units=hidden_unit,
                model_class=model_class
               )
    mean_acc = 0
    if format.lower() == '.pt':
        mean_acc = EvaluateOnTest_pt(model=loaded_model, 
                                    data_loader=test_loader,  
                                    device=device)
    elif format.lower() == '.onnx':
        mean_acc = EvaluateOnTest_onnx(ort_session=loaded_model, 
                                    data_loader=test_loader)
    else:
        raise ValueError(f"Unsupported format '{format}'. Please use '.pt' or '.onnx'.")
    m = f"""
    *Evaluation on Complete Unseen Data
    ---- Test Score ----
    Model Name : {model_name}
    Model Format : {format}
    Model Location : {model_path}

    Mean Accuracy : {(mean_acc*100):.2f}%
    ---- END ----

"""
    print(m)
if __name__ == '__main__':
    main()