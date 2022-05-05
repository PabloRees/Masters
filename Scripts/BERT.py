#Reference: https://www.youtube.com/watch?v=Osj0Z6rwJB4&ab_channel=VenelinValkov
from inspect import currentframe, getframeinfo

def getLineNo():
    cf = currentframe()
    return cf.f_back.f_lineno

print(f'line {getLineNo()}')
import transformers
print(f'line {getLineNo()}')
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
print(f'line {getLineNo()}')
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from R_Interface import *
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/heavy_final_dataset(101566, 7).csv') #need to decide on a dataset to read in here

sample_txt = 'I love to play fifa with Obama. Putin is a warlord.'

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict = False) #can use cased or uncased - refers to Case of the words
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

###Special Tokens
tokenizer.sep_token, tokenizer.sep_token_id
tokenizer.cls_token, tokenizer.cls_token_id
tokenizer.pad_token, tokenizer.pad_token_id
tokenizer.unk_token, tokenizer.unk_token_id

encoding = tokenizer.encode_plus(
    sample_txt,
    max_length = 32,
    add_special_tokens=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='pt'
)

encoding.keys()

encoding['input_ids']
encoding['attention_mask']

###Choosing sequence length

token_lens = []

for txt in df.content:
    tokens = tokenizer.encode(txt, max_length=512) #need to choose an appropriate length for the speeches. Might just be able to drop max_length param
    token_lens.append(len(tokens))

sns.distplot(token_lens)

class GPReviewDataset(Dataset):

    def __init__(self,review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])

        encoding = tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return {
            'review_text' : review,
            'input_ids' : encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(self.target[item], dtype=torch.long)

        }

MAX_LEN = 128 #needs to be specified for speeches
BATCH_SIZE = 8
EPOCHS = 20
RANDOM_SEED = 42


df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED) #still need to import df
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

df_train.shape, df_test.shape, df_val.shape

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        review=df.content.to_numpy(),
        targets = df.y_cats, #these y_cats need to be the categorized logdif_date_resids
        tokenizer = tokenizer,
        max_len=max_len
    )

    return  data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME) #need to create the 'PRE_TRAINED_MODEL_NAME' variable still

last_hidden_state, pooled_output =  bert_model(
    input_ids = encoding['input_ids'],
    attention_mask = encoding['attention_mask'],
    )

###Building a Sentiment Classifier

class SentimentClassifier(nn.Module):
    def __init__(self,n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.output = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


model = SentimentClassifier(len(class_names)) #need to specify class_names
model = model.to(device) #need to specify GPU device

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

###Training using hugging face and bert

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses= []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses=[]
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:

            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
        )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_acc'].append(val_acc)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'model.bin')
        best_accuracy = val_acc




























