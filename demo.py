import ImageBind.data as data
import llama
import json
import torch
import pandas as pd
import cv2
import fundus_prep as prep
# import BERTSimilarity.BERTSimilarity as bertsimilarity
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

def capitalize_sentences(text):
    # if not text[0].isalpha():
    #     text = text[1:]
    sentences = text.split('. ')  # Split the text into sentences

    capitalized_sentences = []
    for sentence in sentences:
        capitalized_sentence = sentence.capitalize()  # Capitalize the first letter
        capitalized_sentences.append(capitalized_sentence)

    capitalized_text = '. '.join(capitalized_sentences)  # Join the sentences back into text
    return capitalized_text

def metrics(preds, labels):
    t_list = [0.5]
    for t in t_list:
        predictions = preds.tolist()
        groundTruth = labels.tolist()
        confusion = confusion_matrix(groundTruth, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        acc = accuracy_score(groundTruth, predictions)
        kappa = cohen_kappa_score(groundTruth, predictions)
        
        from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score
        precision = TP / float(TP+FP)
        sensitivity = TP / float(TP+FN)
        specificity = TN / float(TN+FP)
        F1 = f1_score(groundTruth, predictions)
        balanced_accuracy = balanced_accuracy_score(groundTruth, predictions)
        
        if np.array(groundTruth).max() > 1:
            auc = roc_auc_score(labels, preds, multi_class='ovr')
        else:
            auc = roc_auc_score(labels, preds)
        # print(list(groundTruth), list(predictions))
        print('Threshold:%.4f\tAccuracy:%.4f\tBalanced_Accuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f' % (
            t, acc, balanced_accuracy, sensitivity, specificity, precision, F1, auc, kappa))
        print('TN: %d\t FN:%d\t TP: %d\t FP: %d\n' % (TN, FN, TP, FP))
        return acc, sensitivity, specificity, precision,  F1, auc, kappa

## CUDA_VISIBLE_DEVICES=0,1
def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    448, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            # try:
            #     img = prep.imread(fopen)
            #     r_img, borders, mask, r_img = prep.process_without_gb(img,img)
            #     image = Image.fromarray(cv2.cvtColor(r_img,cv2.COLOR_BGR2RGB))
            # except:
            #     image = Image.open(fopen).convert('RGB')

            # try:
            #     bbox = image.getbbox()
            #     image = image.crop(bbox)
            # except:
            #     pass
            # image = Image.open(filename).convert('HSV')
            image = ImageEnhance.Contrast(image)
            image = image.enhance(1.3)
            image = np.array(image)
            min_R = np.min(image[:,:,0])
            min_G = np.min(image[:,:,1])
            min_B = np.min(image[:,:,2])
            image[:,:,0] = image[:,:,0] - min_R +1
            image[:,:,1] = image[:,:,1] - min_G +1
            image[:,:,2] = image[:,:,2] - min_B +1
            image = Image.fromarray(image.astype('uint8')).convert('HSV')

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)

llama_dir = "/llama-adapter/llama_model_weights" # /path/to/LLaMA/

device = "cuda" if torch.cuda.is_available() else "cpu"
model = llama.load("/output_dir/checkpoint-12_30-20.pth", llama_dir).to(device)
model.eval()

model.to(device)

ann = json.load(open('/Dataset/-MLLM-Fundus/All_json/RITE40_Test20.json'))
fileHandler  =  open("/Valset_list_updated.txt",  "r")
listOfLines  =  fileHandler.readlines()
for  line in  listOfLines:
    ann += json.load(open(line.strip()))

Batchsize= 8
predict = []
GroundTruth = []
image = []
similarity = []
cls_labels = []
cls_preds = []
instruction_gt = []
Keyword_list = []
data_dict = {}

# bertsimilarity=bertsimilarity.BERTSimilarity()
for index in range(0, len(ann), Batchsize):
    sample = []
    prompt = []
    prompt_gt = []
    label = []
    for i in range(Batchsize):
        try:
            url = ann[index+i]['ImageID']
            # data_dict[url] = data_dict.get(url, 0) + 1
            sample.append(ann[index+i]['ImageID'])
            prompt_gt.append(ann[index+i]['Instruction'])
            prompt.append(llama.format_prompt(ann[index+i]['Instruction']))
            label.append(ann[index+i]['Answer'])

            sample.append(ann[index+i]['ImageID'])
            prompt_gt.append(ann[index+i]['Instruction0'])
            prompt.append(llama.format_prompt(ann[index+i]['Instruction0']))
            label.append(ann[index+i]['Answer0'])

            sample.append(ann[index+i]['ImageID'])
            prompt_gt.append(ann[index+i]['Instruction1'])
            prompt.append(llama.format_prompt(ann[index+i]['Instruction1']))
            label.append(ann[index+i]['Answer1'])

            if ann[index+i]['Keyword'][0] == 'A':
                cls_labels.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if 'FHE_label' in ann[index+i].keys():
                    if float(ann[index+i]['FHE_label']) > 0:
                        cls_labels[len(cls_labels)-1][1] = 1.0
                        cls_labels[len(cls_labels)-1][0] = 0.0
                if 'OCD_label' in ann[index+i].keys():
                    if float(ann[index+i]['OCD_label']) > 0:
                        cls_labels[len(cls_labels)-1][2] = 1.0
                        cls_labels[len(cls_labels)-1][0] = 0.0
                if 'FBC_label' in ann[index+i].keys():
                    if float(ann[index+i]['FBC_label']) > 0:
                        cls_labels[len(cls_labels)-1][3] = 1.0
                        cls_labels[len(cls_labels)-1][0] = 0.0
                if 'Macular_label' in ann[index+i].keys():
                    if float(ann[index+i]['Macular_label']) > 0:
                        cls_labels[len(cls_labels)-1][4] = 1.0
                        cls_labels[len(cls_labels)-1][0] = 0.0
                if 'AV_label' in ann[index+i].keys():
                    if float(ann[index+i]['AV_label']) > 0:
                        cls_labels[len(cls_labels)-1][5] = 1.0
                        cls_labels[len(cls_labels)-1][0] = 0.0
                cls_labels.append(cls_labels[len(cls_labels)-1])
                cls_labels.append(cls_labels[len(cls_labels)-1])
                Keyword_list.append(ann[index+i]['Keyword'])
                Keyword_list.append(ann[index+i]['Keyword'])
                Keyword_list.append(ann[index+i]['Keyword'])
            else:
                cls_labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                cls_labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                cls_labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                Keyword_list.append(ann[index+i]['Keyword'])
                Keyword_list.append(ann[index+i]['Keyword'])
                Keyword_list.append(ann[index+i]['Keyword'])
            # if ann[index+i]['Keyword'][0] == 'A':
            #     cls_labels.append(1.0)
            #     if 'AVR_label' in ann[index+i].keys():
            #         if float(ann[index+i]['AVR_label']) > 0:
            #             cls_labels[index+i] = 2.0
            #     if 'CDR_label' in ann[index+i].keys():
            #         if float(ann[index+i]['CDR_label']) > 0:
            #             cls_labels[index+i] = 3.0
            #     if 'Glaucoma_label' in ann[index+i].keys():
            #         if float(ann[index+i]['Glaucoma_label']) > 0:
            #             cls_labels[index+i] = 4.0
            #     if 'Myopia_label' in ann[index+i].keys():
            #         if float(ann[index+i]['Myopia_label']) > 0:
            #             cls_labels[index+i] = 5.0
            #     if 'DR_label' in ann[index+i].keys():
            #         if float(ann[index+i]['DR_label']) > 0:
            #             cls_labels[index+i] = 6.0
            #     if 'Macular_label' in ann[index+i].keys():
            #         if float(ann[index+i]['Macular_label']) > 0:
            #             cls_labels[index+i] = 7.0
            # else:
            #     cls_labels.append(0.0)
        except:
            continue
    # print(sample)
    input = load_and_transform_vision_data(sample, device)
    # prompt = prompt.to(device)
    results, cls_pred = model.generate(input, prompt, input_type="vision")
    for i in range(len(results)):
        # print(results[i])
        # print(len(label[i]))
        # print(len(results[i]))
        # similarity.append(bertsimilarity.calculate_distance(label[i],results[i]))
        # similarity.append(0)
        image.append(sample[i])
        instruction_gt.append(prompt_gt[i])
        GroundTruth.append(label[i])
        index = results[i].rfind('.')
        if (index+1)!=len(results[i]):
            results[i] = results[i][:index+1]
        predict.append(capitalize_sentences(results[i]).replace(' i ', ' I ').replace(' i\'', ' I\''))
        # cls_labels.append(cls_labels[i])
        # cls_preds.append([cls_pred[i][0].cpu().numpy(),cls_pred[i][1].cpu().numpy(),cls_pred[i][2].cpu().numpy(),cls_pred[i][3].cpu().numpy(),cls_pred[i][4].cpu().numpy(),cls_pred[i][5].cpu().numpy()])
        cls_preds.append(cls_pred[i])

    dict = {'ImageID': image, 'instruction_gt': instruction_gt, 'GT': GroundTruth , 'Predict': predict, 'Keyword': Keyword_list, 'cls_labels': cls_labels,'cls_preds': cls_preds}
    df = pd.DataFrame(dict)
    df.to_csv('./results/Test_result_12_30-20.csv',index=False)

assert len(cls_labels) == len(cls_preds)

cls_labels = np.array(cls_labels)
cls_preds = np.array(cls_preds)

class_1 = cls_labels[:,0]
class_2 = cls_labels[:,1]
class_3 = cls_labels[:,2]
class_4 = cls_labels[:,3]
class_5 = cls_labels[:,4]
class_6 = cls_labels[:,5]
class_1_pred = cls_preds[:,0]
class_2_pred = cls_preds[:,1]
class_3_pred = cls_preds[:,2]
class_4_pred = cls_preds[:,3]
class_5_pred = cls_preds[:,4]
class_6_pred = cls_preds[:,5]

metrics(class_1_pred, class_1)
metrics(class_2_pred, class_2)
metrics(class_3_pred, class_3)
metrics(class_4_pred, class_4)
metrics(class_5_pred, class_5)
metrics(class_6_pred, class_6)
