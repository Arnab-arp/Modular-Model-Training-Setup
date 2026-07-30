from torchvision import transforms

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomAutocontrast(p=0.1),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(std=[0.229, 0.224, 0.225], # Scales 0-255 down to 0.0-1.0
                         mean=[0.485, 0.456, 0.406]) # Shifts distribution based on 0.0-1.0 range
])

VALIDATION_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(std=[0.229, 0.224, 0.225],
                        mean=[0.485, 0.456, 0.406])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(std=[0.229, 0.224, 0.225],
                        mean=[0.485, 0.456, 0.406])
])
