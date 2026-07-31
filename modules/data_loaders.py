import os
import gc
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from modules.data_trainsforms import(
    TRAIN_TRANSFORM,
    TEST_TRANSFORM,
    VALIDATION_TRANSFORM
)

def LoadDataFromPath(train_path, val_path, test_path, batch_size=32, verbose=True):
    workers = os.cpu_count()
    train_loader, val_loader, test_loader = None, None, None
    classes, cls2idx = None, None
    input_shape, output_shape = None, None
    try:
        train_data = ImageFolder(root=train_path,
                                transform=TRAIN_TRANSFORM,
                                target_transform=None)
        val_data = ImageFolder(root=val_path,
                            transform=VALIDATION_TRANSFORM,
                            target_transform=None)
        test_data = ImageFolder(root=test_path,
                            transform=TEST_TRANSFORM,
                            target_transform=None)

        classes = train_data.classes
        cls2idx = train_data.class_to_idx

        train_loader = DataLoader(dataset=train_data,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=workers)
        val_loader = DataLoader(dataset=val_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=workers)
        test_loader = DataLoader(dataset=test_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=workers)

        first_batch_images = next(iter(train_loader))[0]
        input_shape = list(first_batch_images.shape)[1] 
        output_shape = len(classes)

        if verbose:
            print("Data Loaded")
            m = f"""
    ---- Sats ----
[Class Info]
Classes : {len(classes)}
Class Names : {classes}
Class 2 Index : {cls2idx}

[Size Info]
Batch Size : {batch_size}
Train Loader Size : {len(train_loader)}
Valid Loader Size : {len(val_loader)}
Test Loader Size : {len(test_loader)}

Image Shape : {list(next(iter(train_loader))[0].shape)}
    ---- END ----
"""     
            print(m)
        del workers, train_data, val_data, test_data
        gc.collect()
    except Exception as e:
        print(f"Failed Loading Data from Paths\n{e}")
    finally:
        return train_loader, val_loader, test_loader, input_shape, output_shape, classes, cls2idx


class TransformSubset(Dataset):
    """
    As the directory is not splitted pror to the training, we have to create a
    Dataset wrapper. So that when the dataset is passed through the DataLoader,
    the image transforms on the fly.
    """
    def __init__(self, data_subset, transform=None):
        self.subset = data_subset
        self.transform = transform
        # self.classes, self.class_to_idx = self._find_classes()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        X, y =  self.subset.dataset[self.subset.indices[index]]
        if self.transform:
            X = self.transform(X)
        return X, y

def LoadDataSplit(data_dir, ratio=(0.75, 0.15, 0.1), batch_size=32, verbose=True):
    workers = os.cpu_count()
    train_loader, val_loader, test_loader = None, None, None
    classes, cls2idx = None, None
    input_shape, output_shape = None, None

    try:
        full_data = ImageFolder(root=data_dir, transform=None)

        train_set, val_set, test_set = random_split(dataset=full_data, 
                                                    lengths=ratio)

        train_data = TransformSubset(data_subset=train_set, transform=TRAIN_TRANSFORM)
        val_data = TransformSubset(data_subset=val_set, transform=VALIDATION_TRANSFORM)
        test_data = TransformSubset(data_subset=test_set, transform=TEST_TRANSFORM)

        classes = full_data.classes
        cls2idx = full_data.class_to_idx

        train_loader = DataLoader(dataset=train_data,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=workers)
        val_loader = DataLoader(dataset=val_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=workers)
        test_loader = DataLoader(dataset=test_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=workers)
        
        first_batch_images = next(iter(train_loader))[0]
        input_shape = list(first_batch_images.shape)[1] 
        output_shape = len(classes)

        if verbose:
            print("Data Loaded")
            m = f"""
    ---- Sats ----
[Class Info]
Classes : {len(classes)}
Class Names : {classes}
Class 2 Index : {cls2idx}

[Size Info]
Batch Size : {batch_size}
Train Loader Size : {len(train_loader)}
Valid Loader Size : {len(val_loader)}
Test Loader Size : {len(test_loader)}

Image Shape : {list(next(iter(train_loader))[0].shape)}
    ---- END ----
"""     
            print(m)
            del workers, train_data, val_data, test_data
            gc.collect()
    except Exception as e:
        print(f"Failed Loading Data from Paths\n{e}")
    finally:
        return train_loader, val_loader, test_loader, input_shape, output_shape, classes, cls2idx

if __name__ == "__main__":
    ds_pth = r"C:\Users\Arnab\Desktop\Small"
    train_loader, val_loader, test_loader, input_shape, output_shape, classes, cls2idx = LoadDataSplit(data_dir=ds_pth, batch_size=1)
