# VisionUnite
This repository is the official implementation of the paper "VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge". The dataset we use for fine-tuning is the [MMFundus](https://github.com/HUANGLIZI/MMFundus) dataset.

![image](https://github.com/HUANGLIZI/VisionUnite/blob/main/VisionUnite_Manuscript.jpg)
**(a)** Previous vision models could only diagnose specific diseases as positive or negative, lacking the ability to provide clinical explanations or interact with patients. However, our proposed VisionUnite changes this approach. It can predict a wide range of diseases and allows real-time conversations with patients, incorporating their feedback. Additionally, VisionUnite offers clear clinical explanations in its output, making it more understandable and useful. **(b)** The label distribution of the proposed MMFundus dataset, which includes eight main categories excluding the "Others" class. **(c)** VisionUnite is built with a transformer-based vision encoder and a specialized vision adapter designed for classifying six different signs including Vascular, Macular, FBC (Fundus Boundary Color), OCD (Optical Cup Disc), FHE (Fundus Hemorrhages Examination), and Other. It includes a vision projector to align visual embeddings with text tokens. **(d)** The illustration of image-text contrastive learning (CLIP Loss). **(e)** The illustration of classification supervised learning (CLS Loss). **(f)** The illustration of text-generation supervised learning (LLM Loss).

## Requirements
Python == 3.8 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage

### 1. Training

You can train to get your own model.

```angular2html
bash ./exps/train.sh
```

### 2. Evaluation

#### Test the Model

Prepare the test data and run the following command
```angular2html
python demo.py
```
