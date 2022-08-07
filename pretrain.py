import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, TrainingArguments, pipeline, AutoConfig
from transformers import RobertaTokenizer, T5ForConditionalGeneration

#チュートリアルここから
os.environ["WANDB_DISABLED"] = "true"


pd.options.display.width = 180
pd.options.display.max_colwidth = 120

#BERT_PATH = "../../input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased"

data_dir = Path('../input/')


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


paths_train = list((data_dir / 'train').glob('*.json'))
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]
df = (
    pd.concat(notebooks_train)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

# Get an example notebook
nb_id = df.index.unique('id')[6]
nb = df.loc[nb_id, :]

df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list


cell_order = df_orders.loc[nb_id]

def get_ranks(base, derived):
    return [base.index(d) for d in derived]

cell_ranks = get_ranks(cell_order, list(nb.index))
nb.insert(0, 'rank', cell_ranks)


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)


ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')

df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])

dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))


#チュートリアルここまで

#再事前学習ここから

import numpy as np
import pandas as pd
import os
import re
# import fasttext
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

    
def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message) for message in df.source]

df.source = df.source.apply(preprocess_text)


#stemmingなどのデータ加工ここまで

from tqdm import tqdm
import sys, os
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModel

if not os.path.exists('text.txt'):
  with open('text.txt','w') as f:
    for id, item in tqdm(df.groupby('id')):
      df_markdown =  item[item['cell_type']=='markdown']
      for source, rank in df_markdown[['source', 'rank']].values:
        cell_source = df_markdown[df_markdown['rank']==(rank+1)]
        if len(cell_source):
          setence = source + ' [SEP] ' + cell_source.source.values[0]
          f.write(setence+'\n')
      

# Train a tokenizer
import tokenizers
from transformers import BertTokenizer, LineByLineTextDataset


#tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

#model = AutoModelForMaskedLM.from_pretrained('microsoft/codebert-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').encoder


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = './text.txt',
    block_size = 512  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset

training_args = TrainingArguments(
    output_dir='/codet5-base-mlm',
    overwrite_output_dir=True,
    evaluation_strategy = 'no',
    gradient_accumulation_steps=1,
    per_device_train_batch_size=64,
    
    save_strategy='epoch',
    
    num_train_epochs=10,

    lr_scheduler_type='cosine',
    weight_decay=0.01,
    warmup_ratio=0.2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

#モデルの保存、ついでにtrainig-argsも保存
trainer.save_model('mlm_model_from_trainer')


#モデルの保存
trainer.model.save_pretrained('mlm-model')


#トークないざーの保存
tokenizer.save_pretrained('model_tokenizer')
config = AutoConfig.from_pretrained('Salesforce/codet5-base')
config.save_pretrained('model_tokenizer')
