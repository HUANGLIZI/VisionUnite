import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from PIL import Image
from PIL import ImageEnhance
Image.LOAD_TRUNCATED_IMAGES = True
from llama.llama_adapter import LLaMA_adapter

import random
import deepspeed
import argparse
import datetime
import json
import numpy as np
import os
import time
import cv2
from pathlib import Path

from engine_pretrain import train_one_epoch, val_one_epoch

from llama import Tokenizer
import fundus_prep as prep
import copy
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

OPTION_DICT = {
    0: "A. normal",
    1: "B. other diseases",
    2: "C. arteriovenous ratio",
    3: "D. cup-to-disc ratio",
    4: "E. glaucoma",
    5: "F. myopia",
    6: "G. diabetic retinopathy",
    7: "H. age-related macular degeneration"
    }

class CaptionCOCO(Dataset):
    def __init__(self, transform, max_words=512, partition='train', tokenizer_path=None):
        # ann = json.load(open("/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/lizihan/MLLM_Dataset/imagebank/imagebank_Clean.json"))
        # ann += json.load(open("/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/lizihan/PMC-OA-Full/PMC_Clean.json"))
        if partition == 'train':
            # self.ann = ann
            ann = json.load(open("/cpfs01/user/lizihan/Dataset/-MLLM-Fundus/All_json/GPT_RITE40_Train20.json"))
            fileHandler  =  open  ("/cpfs01/user/lizihan/Trainset_list_updated.txt",  "r")
            listOfLines  =  fileHandler.readlines()
            for  line in  listOfLines:
                ann += json.load(open(line.strip()))
            self.ann = ann
            self.data_dict = {}
            
        else:
            ann = json.load(open("/cpfs01/user/lizihan/Dataset/-MLLM-Fundus/All_json/GPT_RITE40_Test20.json"))
            fileHandler  =  open  ("/cpfs01/user/lizihan/Valset_list_updated.txt",  "r")
            listOfLines  =  fileHandler.readlines()
            for  line in  listOfLines:
                ann += json.load(open(line.strip()))
            self.ann = ann
            self.data_dict = {}
            # sample_num = 10000
            # self.ann = random.sample(ann, sample_num)

        self.transform = transform
        self.max_words = max_words
        self.max_keywords = 32
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer1 = tokenizer
    def __len__(self):
        return len(self.ann)

    def get_qa(self, orig_qa):
        qa = orig_qa.replace('\n\n', '\n')
        qa_list = qa.split('\n')
        qa_list = [sentence[6:] for sentence in qa_list]
        return qa_list

    def __getitem__(self, index):
        labels_cls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data_item = self.ann[index]
        if data_item['Keyword'][0] == 'A':
            labels_cls[0] = 1.0
            if 'FHE_label' in data_item.keys():
                if float(data_item['FHE_label']) > 0:
                    labels_cls[1] = 1.0
                    labels_cls[0] = 0.0
            if 'OCD_label' in data_item.keys():
                if float(data_item['OCD_label']) > 0:
                    labels_cls[2] = 1.0
                    labels_cls[0] = 0.0
            if 'FBC_label' in data_item.keys():
                if float(data_item['FBC_label']) > 0:
                    labels_cls[3] = 1.0
                    labels_cls[0] = 0.0
            if 'Macular_label' in data_item.keys():
                if float(data_item['Macular_label']) > 0:
                    labels_cls[4] = 1.0
                    labels_cls[0] = 0.0
            if 'AV_label' in data_item.keys():
                if float(data_item['AV_label']) > 0:
                    labels_cls[5] = 1.0
                    labels_cls[0] = 0.0
        
        if 'ImageID' in data_item.keys():
            url = data_item['ImageID']
            # self.data_dict[url] = self.data_dict.get(url, 0) + 1
            question = data_item['Instruction'] # + " Please choose from the following options: A. normal, B. other diseases, C. arteriovenous ratio, D. cup-to-disc ratio, E. glaucoma, F. myopia, G. diabetic retinopathy, H. age-related macular degeneration."
            answer = data_item['Answer'] # + " The answer is " + OPTION_DICT[int(labels_cls[0])] + "."
            if 'Instruction0' in data_item.keys():
                question0 = data_item['Instruction0']
                answer0 = data_item['Answer0']
                question1 = data_item['Instruction1']
                answer1 = data_item['Answer1']
            else:
                question0 = " "
                answer0 = " "
                question1 = " "
                answer1 = " "
            Keyword = data_item['Keyword']
            
            filename = url
            
            image = Image.open(filename).convert('RGB')
            
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
            image = self.transform(image)
            # input1 = question
            question = {'instruction': question}
            question0 = {'instruction': question0}
            question1 = {'instruction': question1}
        elif 'image' in data_item.keys():
            url = data_item['image']
            question = "This is a "+ data_item['modality']+ " image. "+data_item['question']
            answer = data_item['caption']
            filename = "/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/lizihan/PMC-OA-Full/images_full/image_only/" + url
            try:
                image = Image.open(filename).convert('RGB')
            except:
                filename = "/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/lizihan/PMC-OA-Full/images_full/image_duplicate/" + url
                image = Image.open(filename).convert('RGB')
            #           image = cv2.imread(filename)
            #           image = Image.fromarray(image)
            image = self.transform(image)
            input1 = {'instruction': question}
        elif 'img_id' in data_item.keys():
            url = data_item['img_id']
            question = "This is a "+ data_item['modality']+ " image. "+data_item['question']
            answer = data_item['answer']
            filename = "/cpfs01/shared/Gvlab-A100/Gvlab-A100_hdd/lizihan/MLLM_Dataset/imagebank/imagebank_image/" + url +'.jpg'
            try:
                image = Image.open(filename).convert('RGB')
                image = self.transform(image)
                input1 = {'instruction': question}
            except:
                print('Truncated image: '+str(data_item['img_id']))
            #           image = cv2.imread(filename)
            #           image = Image.fromarray(image)            
        else:
            image = torch.zeros(3, 448, 448)
            # input1 = {'instruction': data_item['instruction'], 'input': data_item['input']}
            input1 = data_item['instruction']
            Keyword = data_item['Keyword']
            answer = data_item['output']
        
        # input1 = PROMPT_DICT['prompt_no_input'].format_map(input1)
        question = PROMPT_DICT['prompt_no_input'].format_map(question)
        question0 = PROMPT_DICT['prompt_no_input'].format_map(question0)
        question1 = PROMPT_DICT['prompt_no_input'].format_map(question1)
        input2 = question + answer + question0 + answer0 + question1 + answer1
        question_index = torch.tensor(self.tokenizer1.encode(question, bos=True, eos=False), dtype=torch.int64)
        answer_index = torch.tensor(self.tokenizer1.encode(question+answer, bos=True, eos=False), dtype=torch.int64)
        question0_index = torch.tensor(self.tokenizer1.encode(question+answer+question0, bos=True, eos=False), dtype=torch.int64)
        answer0_index = torch.tensor(self.tokenizer1.encode(question+answer+question0+answer0, bos=True, eos=False), dtype=torch.int64)
        question1_index = torch.tensor(self.tokenizer1.encode(question+answer+question0+answer0+question1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer1.encode(input2, bos=True, eos=True), dtype=torch.int64)
        
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
            
        labels = copy.deepcopy(input2)

        labels[:len(question_index)] = -1
        labels[len(answer_index):len(question0_index)] = -1
        if len(question1_index) >= self.max_words:
            labels[len(answer0_index):] = -1
        else:
            labels[len(answer0_index):len(question1_index)] = -1

        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        
        # return input2, labels, input2_mask, image
        return input2, labels, labels_cls, image, Keyword

def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train') # llama-adapter: currently no use
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use_checkpoint', default=False, type=bool)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = "7B"
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(llama_ckpt_dir, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.get_trainable_params().items()])
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    # optionally resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # create data
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=(448, 448), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                     antialias=None),  # 3 is bicubic
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_train = CaptionCOCO(transform=transform_train, max_words=512, partition='train', tokenizer_path=llama_tokenzier_path)
    dataset_val = CaptionCOCO(transform=transform_train, max_words=512, partition='val', tokenizer_path=llama_tokenzier_path)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # if args.output_dir and ((epoch+1) % 5 == 0 or (epoch + 1) == args.epochs):
        if args.output_dir and ((epoch+1) % 5 == 0 ):

            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
