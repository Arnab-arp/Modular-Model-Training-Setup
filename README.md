
# Pytorch Modular Training Pipeline

This project is built to train Computer Vision models as effortlessly as possible. With easy to add custom models and registering it, this repo helps by reducing the amount of code written for a training and evaluation Pipeline, including saving pytorch models in both **.pt** and **.onnx** formats.
----------------------------------------------------------




## A. Features

1. **Pre-built Models**: This repo alredy contains **3** pre-built models.
    
    **(1.a) TinyVGG model** : This model is built on a small scale VGG model explained in [CNN Explainer](https://poloclub.github.io/cnn-explainer/).

    **(1.b) VGG16 model** : Built entirely from online refence of architecture. [Viso.ai](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)

    **(1.c) Resnet50 model** : Model build entirely from the research paper. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

2. **OS Support**: This repo is made compatible to support both **Linux** and **Windows** machines.

3. **Easy Model Registration**: This repo also supports **Custom model integration** and registering model so that it can be used for any custom model integration. [Refer to docs]()

4. **Dynamic Input and Output shapes**: Conficts such as mismatch in Input shapes and output shapes has been solved by taking the Input-Output shapes directly from DataLoaders

5. **Support Google Colab Instance**: Despite running locally, this pipeline can also be run on **Google Colab sessions**. Plese refer to [Notebook](https://github.com/Arnab-arp/Modular-Model-Training-Setup/blob/master/06_pytorch_going_modular_modular.py)

6. **Support for raw datasets**: If the dataset provided is **Raw** but structured in classes, the pipeline introduces random splits to convert to **train - test - val** splits automatically. [Refer to docs]()

```
Standard Image Classification directory structure

dataset
    |-> Class 1
    |      |-> img1.jpg
    |      |-> img2.jpg
    |     ...  ...
    |      |-> imgN.jpg
    |-> Class 2
    |      |-> img1.jpg
    |      |-> img2.jpg
    |     ...  ...
    |      |-> imgN.jpg
   ...
    |-> Class N
           |-> img1.jpg
           |-> img2.jpg
          ...  ...
           |-> imgN.jpg
```
## B. Setup
Setting up this repo is as easy as it gets

1. Clone the repo, and change directory
```bash
git clone https://github.com/Arnab-arp/Modular-Model-Training-Setup.git
cd Modular-Model-Training-Setup
```
2. Create Virtual environment
- Python Command (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
```
OR 
```bash
uv venv .venv
.venv\Scripts\activate
``` 
- Python Command (Linux)
```bash
sudo apt update && sudo apt install python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
```
OR
```bash
uv venv .venv
source .venv/bin/activate
```
**NOTE** For using UV commands please install UV first. [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

3. Install the requirements from `requirements.txt`
- Windows
```bash
pip install -r requirements.txt
```
OR
```bash
uv pip install -r requirements.txt
```
- Linux
```bash
pip3 install -r requirements.txt
```
OR
```bash
uv pip install -r requirements.txt
```

4. Run the file
- Windows 
```bash
python train_model.py --help
```
OR
```bash
uv run train_model.py --help
```
- Linux 
```bash
python3 train_model.py --help
```
OR
```bash
uv run train_model.py --help
```


## Learner tips
- [Understanding Tesors - Dan Fleisch](https://www.youtube.com/watch?v=f5liqUk0ZTw)
- [NN from Scratch](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
- [Daniel Bourke](https://www.youtube.com/channel/UCr8O8l5cCX85Oem1d18EezQ/videos)
- [Pytorch - FreeCodeCamp](https://www.youtube.com/watch?v=GIsg-ZUy0MY)
