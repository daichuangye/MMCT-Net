# MMCT-Net


This repo is the official implementation of "**Multi-Modal Convolution Transformer Network in Medical Image Segmentation**"

## Requirements

Python == 3.7.2 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version we use is 1.17.5. We can install bert-embedding first, and install NumPy then.

## Usage

### 1. Data Preparation
#### 1.1. QaTa-COV19 and BUSI Datasets
The original data can be downloaded in following links:
* QaTa-COV19 Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* BUSI Dataset - [Link (Original)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

  *(Note: The text annotation of QaTa-COV19 train and val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDkm5jsTw5dX_RpuRr?e=uaZq6W).
  The partition of train set and val set of QaTa-COV19 dataset [download link](https://1drv.ms/u/s!AihndoV8PhTDgt82Do5kj33mUee33g?e=kzWl8y).
  The text annotation of QaTa-COV19 test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDkj1vvvLt2jDCHqiM?e=d5d2hc).)*
  
  *(Note: The text annotation of BUSI train datasets [download link](https://docs.google.com/spreadsheets/d/1_AzOgBZKzgG8E3_LvXR1_wxCPIO4n5I-/edit?usp=sharing&ouid=105099223925277018244&rtpof=true&sd=true).
  The text annotation of BUSI val datasets [download link](https://docs.google.com/spreadsheets/d/1JDpF_EVxOGt5u-hIiSevIBeXQGNe0Np5/edit?usp=sharing&ouid=105099223925277018244&rtpof=true&sd=true).)*
  

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-Covid19
    │   ├── Train_Folder
    |   |   ├── Train_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    |	    ├── Val_text.xlsx
    │       ├── img
    │       └── labelcol
```



### 2. Training

#### 2.1. Pre-training
You can replace LVIT with U-Net for pre training and run:
```angular2html
python train_model.py
```

#### 2.2. Training

You can train to get your own model. It should be noted that using the pre-trained model in the step 2.1 will get better performance or you can simply change the model_name from LViT to LViT_pretrain in config.

```angular2html
python train_model.py
```

### 3. Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc.


## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)

Our code is modified and adapted on these great repositories:
* [LViT](https://github.com/HUANGLIZI/LViT)
