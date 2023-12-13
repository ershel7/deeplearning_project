import os
import torch
import pandas as pd
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import csv
from keras.preprocessing.sequence import pad_sequences
# kobert
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
model = AutoModel.from_pretrained("monologg/kobert").cuda()

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)

cuda = torch.device('cuda')

# read_raw_data
DATASET_DIR = './data/'
SAVE_DIR = './'
X = pd.read_csv(os.path.join(
    DATASET_DIR, 'dataset.csv'), encoding='utf-8')
X.head()

# essay_data_first_fix
essay_data_origin = X['ESSAY_CONTENT']
essay_data = []

#   holistic
for data in essay_data_origin:
    new = data.replace('<span>', '').replace(
        '</span>', '').replace('\n', '').replace('\t', '')
    essay_data.append(new)


def essay_to_wordlist(essay_v):
    """Remove the tagged labels and word tokenize the sentence."""
    token = sp(essay_v)
    return token


def essay_to_sentences(essay_v):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    raw_sentences = essay_v.split('#@문장구분#')
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences = essay_to_wordlist(raw_sentence)
            if tokenized_sentences != []:
                sentences.append(tokenized_sentences)
    return sentences


# tokenized essays
essays = []
for ix, essay in enumerate(essay_data):
    sentences = essay_to_sentences(essay)
    essays.append(sentences)

# # embedding & make file
ff = open(os.path.join(
    DATASET_DIR, 'embedded_features_kobert_holistic.csv'), 'a', newline='')
writer_ff = csv.writer(ff)

sent_max_len = 50
for ix in range(len(essays)):
    inputs = tokenizer.batch_encode_plus(essays[ix])
    ids_new = pad_sequences(inputs['input_ids'],
                            maxlen=sent_max_len, padding='post')
    mask_new = pad_sequences(
        inputs['attention_mask'], maxlen=sent_max_len, padding='post')
    out = model(input_ids=torch.tensor(ids_new).cuda(),
                attention_mask=torch.tensor(mask_new).cuda())
    embedded_features = out[0].detach().cpu()[:, 0, :].numpy()
    for i in embedded_features:
        writer_ff.writerow(i)
    torch.cuda.empty_cache()
ff.close()
