import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
import random
import time
from early_stopping import *

# import wandb
import wandb
api_key = "72c0d357f2f7c92c0ef201aa9adf1f37c9bb13e5"
wandb.login(key=api_key)
wandb_id = f"PL{round(time.time())}" # ID on WandB
group = f'codet5-base-{wandb_id}'
competition = "AI4Code"
_wandb_kernel = "deb"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='Salesforce/codet5-base')
parser.add_argument('--tokenizer_name_or_path', type=str, default='Salesforce/codet5-base')
parser.add_argument('--train_mark_path', type=str, default='data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='data/val_fts.json')
parser.add_argument('--val_path', type=str, default="data/val.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)

args = parser.parse_args()
os.mkdir("./outputs")
os.mkdir("./es_checkpoint")
data_dir = Path('../input')

import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import nltk
nltk.download('wordnet')

#ステミングなどのデータ加工ここから
stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        #return document

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

train_df_mark.source = train_df_mark.source.apply(preprocess_text)
val_df_mark.source = val_df_mark.source.apply(preprocess_text)

order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=args.model_name_or_path, tokenizer_name_or_path=args.tokenizer_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts)
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path,tokenizer_name_or_path=args.tokenizer_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.module.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs):
    
    
    #np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.module.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
    #                                            num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    es = EarlyStopping(patience=6, mode="max")
    
    run = wandb.init(
        project = 'AI4Code',
        job_type = 'Train',
        group = group,
        tags = ["codet5-base", wandb_id],
        name = f'codet5-base-{wandb_id}',
        anonymous='must')
    wandb.watch(model, log_freq = 100)

    for e in range(epochs):       
        model.module.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
            wandb.log({'Train Loss': avg_loss})
           

        y_val, y_pred = validate(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        valid_kendall_loss =  kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", valid_kendall_loss)
        wandb.log({'Eval Loss': valid_kendall_loss})
        torch.save(model.module.state_dict(), "./outputs/model.bin")
        es( valid_kendall_loss,model.module,model_path="./es_checkpoint/model_es.bin")
        if es.early_stop:
            break

    return model, y_pred


model = MarkdownModel(args.model_name_or_path)
model = torch.nn.DataParallel(model, device_ids= [0, 1])
model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
wandb.finish()

