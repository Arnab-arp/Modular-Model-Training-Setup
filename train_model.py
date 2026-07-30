from argparse import ArgumentParser
from modules.model_registry import Registry

def main():
    Registry.discover_models('modules.models')
    all_models = Registry.registered_models()

    parser = ArgumentParser(description="Simple Command-line tool for Model Training.\nType --help or -h for more commands")

    parser.add_argument("-m","--model", metavar='NAME', help=f"(Case Sensitive) Model to train. Models Listed : {list(all_models.keys())}")
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

    model_name = None
    dir_path = None
    train_dir = None
    test_dir = None
    val_dir = None
    lr = None
    batch_size = None
    epoch = None
    verbose = True
    save_name = None
    model_format = None 


    args = parser.parse_args()
    parse_dict = dict(vars(args).items())
    for arg_name, val in vars(args).items():

        if arg_name == 'model':
            model_name = val

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



    # from pathlib import Path
    # from modules.data_loaders import LoadDataFromPath, LoadDataSplit
    # from modules.engine import train_step, eval_step
    # from modules.utility import measure_accuracy, measure_time, PerformanceGraph

if __name__ == '__main__':
    main()