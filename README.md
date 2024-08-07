# VisionUnite
This repository is the official implementation of the paper "VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge". [Arxiv](https://arxiv.org/abs/2408.02865). The dataset we use for fine-tuning is the [MMFundus](https://github.com/HUANGLIZI/MMFundus) dataset.

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

#### 2.1 Test the Model

Prepare the test data and run the following command
```angular2html
python demo.py
```

#### 2.2 Pre-trained models
To obtain pre-trained models for the MMFundus dataset, you can contact the email address zhanli@uw.edu. We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:
```angular2html
Name/Homepage/Google Scholar: (Tell us who you are.)
Primary Affiliation: (The name of your institution or university, etc.)
Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
How to use: (Only for academic research, not for commercial use or second-development.)
```
