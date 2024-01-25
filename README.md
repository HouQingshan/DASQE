# DASQE

This is the implementation of the  "A Collaborative Self-supervised Domain Adaptation for Low-Quality Medical Image Enhancement". 

## Data preparation

1. Firstly, download the datasets from the following links:
   
   * EyeQ Dataset - [Diabetic Retinopathy Detection | Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)
   * Corneal Nerve Dataset - [Link](http://bioimlab.dei.unipd.it/)
   * ISIC-2017 Dataset - [ISIC Challenge](https://challenge.isic-archive.com/data/#2017)
   * Endoscopy Dataset - [Link](https://drive.google.com/file/d/1MuO2SbGgOL)
   * Chest X-ray Dataset - [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC)
   * Cardiac MRI Dataset - [Link](https://www.cardiacatlas.org/amrg-cardiac-atlas/)

2. Split the datasets into train/test.

## Construction of multiple patch domains

1. Serialize the original images via '**patch_serialize.py**'.

2. Initialize high-/low-quality domains $\mathbb{H}$ and $\mathbb{L}$ by clustering algorithm in folder '**./LocalAggregation**'. Next, update $\mathbb{H}$ and $\mathbb{L}$ through '**quality_assessment_scheme.py**'.

3. Construct source style domains $\mathbb{S}$  and target style domains $\mathbb{T}$ from $\mathbb{H}$ through clustering algorithm in folder '**./LocalAggregation**'.

Then prepare the datasets in the following format for easy training: 

（data_l, data_s, and data_t represent patches from $\mathbb{L}$, $\mathbb{S}$ and $\mathbb{T}$, respectively)

```angular2html
├── dataset
│   ├── EyeQ
│   │   ├── Test_Folder
│   │   │   ├── data_l  
│   │   │   ├── data_s
│   │   │   └── data_t
│   │   └── Train_Folder
│   │   │   ├── data_l
│   │   │   ├── data_s
│   │   │   └── data_t
│   ├── Corneal Nerve
│   │   ├── Test_Folder
│   │   │   ├── data_l
│   │   │   ├── data_s
│   │   │   └── data_t
│   │   └── Train_Folder
│   │   │   ├── data_l
│   │   │   ├── data_s
│   │   │   └── data_t
│   ├── ISIC
│   │   ......
│   ├── Endoscopy
│   │   ......
│   ├── Chest X-ray
│   │   ......
│   ├── Cardiac MRI
│   │   ......
```

## Package install

Run

```bash
pip install -r requirements.txt
```

## Training

Change the settings in `configs/unit_noise2clear-bn.yaml` and run the DASQE model

```bash
python train.py --output_path ${LOG_DIR}$
```

## Testing and visualization

Run 

```bash
python test.py --input_a ${Low_quality_path}$ --input_c ${target_style_path}$ --output_folder ${output_image_path}$ --checkpoint ${pretrain_model_path}$ --psnr
```
