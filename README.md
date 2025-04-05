# publish-porfolio_transformer_vnstock
Portfolio optimization with return prediction using Transformer Model
# Guide

## Step 1: Install Anaconda

## Step 2: Run Anaconda

## Step 3: Run cmd create conda environment
```bash
conda create --name tf_gpu
```
## Step 4: Activate Conda
```bash
conda activate tf_gpu
```
## Step 5: Install Pip and install tensorflow
```bash
conda install pip
pip install tensorflow
pip install numpy==1.23.4
pip install keras==2.8
pip install clang==5.0
pip install absl-py==0.10
pip install google-auth==1.6.3
pip install flatbuffers==1.12
pip install scikit-learn==1.3.2
pip install pandas==2.0.2
pip install matplotlib==3.8.0
```

## Step 6: Run script
```bash
conda activate tf_gpu_207
python 0_ExcutedGRUModel.py
python 0_ExcutedLSTMModel.py
python 0_ExcutedRestnetModel.py
python 0_ExcutedTransformerModel.py
```

## Data
https://drive.google.com/drive/folders/1nSq92WPAbYyFtepGnbSeXER3ynUGVFmY?usp=drive_link
