import os
import numpy as np
import pandas as pd
import csv
import torch
import datetime
from pytz import timezone
from kobert.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
# kobert
from transformers import AutoTokenizer, AutoModel
# tensorflow & keras & sklearn
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import keras.utils
import math
from sklearn.metrics import cohen_kappa_score


tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
model = AutoModel.from_pretrained("monologg/kobert").cuda()

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)

cuda = torch.device('cuda')

# read_raw_data
DATASET_DIR = './data/'
SAVE_DIR = './'
X_origin = pd.read_csv(os.path.join(
    DATASET_DIR, 'dataset.csv'), encoding='utf-8')
rubric = pd.read_csv(os.path.join(
    DATASET_DIR, 'rubric.csv'), encoding='cp949')
test_ids = pd.read_csv(os.path.join(
    DATASET_DIR, 'testset.csv'), encoding='cp949')
test_ids_list = test_ids['ID'].tolist()
X = X_origin.iloc[test_ids_list]
X.reset_index(drop=True, inplace=True)

# essay_data_first_fix
essay_data_origin = X['ESSAY_CONTENT']
essay_data = []

# one type
y = []
score_list = []
for i, data in enumerate(essay_data_origin):
    new = data.replace('<span>', '').replace(
        '</span>', '').replace('\n', '').replace('\t', '')
    essay_data.append(new)
    score_list.extend([round(X['exp1'][i]/3, 2), round(X['exp2'][i]/3, 2), round(X['exp3'][i]/3, 2), round(X['org1'][i]/3, 2),
                       round(X['org2'][i]/3, 2), round(X['org3'][i]/3, 2), round(X['org4'][i]/3, 2), round(X['cont1'][i]/3, 2), round(X['cont2'][i]/3, 2), round(X['cont3'][i]/3, 2), round(X['cont4'][i]/3, 2)])
    y.append(score_list)
    score_list = []
y = pd.DataFrame(y)


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

emb_file = os.path.join(DATASET_DIR, 'embedded_features_kobert_inference.csv')
if not os.path.exists(emb_file):
    # # embedding & make file
    ff = open(os.path.join(
        DATASET_DIR, 'embedded_features_kobert_inference.csv'), 'a', newline='')
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

# read_embedded_data
embedded_essay_raw = pd.read_csv(os.path.join(
    DATASET_DIR, 'embedded_features_kobert_inference.csv'), encoding='cp949')
embedded_essay = []
embedded_essay_raw.shape

tmp_ix = 0
for ix, essay_raw in enumerate(essays):
    tmp_len = len(essay_raw)
    essay = embedded_essay_raw[tmp_ix:tmp_ix + tmp_len]
    embedded_essay.append(essay)
    tmp_ix += tmp_len

# data generator


class DataGenerator(keras.utils.all_utils.Sequence):

    def __init__(self, ids, batch_size=64, shuffle=True):
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, index):
        # Generated data containing batch_size samples
        batch_ids = self.ids[index *
                             self.batch_size:(index + 1) * self.batch_size]

        essays = list()
        scores = list()
        for ix in batch_ids:
            essay = embedded_essay[ix]
            score = y.iloc[ix]
            essays.append(essay)
            scores.append(score)
        essays = pad_sequences(
            essays, maxlen=128, padding='pre', dtype='float')

        return np.array(essays), np.array(scores)


# learn
current = datetime.datetime.now(timezone('Asia/Seoul'))
current_time = current.replace(microsecond=0)
print("befort inference : ")
print(current_time)
batch_size = 64

test_gen = DataGenerator(
    range(len(essays)), batch_size=batch_size, shuffle=False)
test_y = y
test_x = np.array(X)
model_name = os.path.join(
    DATASET_DIR, 'kobert_model.h5')
sentence_model = load_model(model_name)

weighted_pred_list = []
weighted_real_list = []

y_sent_pred = sentence_model.predict(test_gen) * 100
y_sent_pred = np.round(y_sent_pred)
y_sent_pred = np.array(y_sent_pred)
y_test = np.array(np.round(test_y.values*100))

for i in range(len(y_test)):
    tmp_rubric = rubric.loc[(
        rubric['SUBJECT'] == test_x[i][2])].to_numpy()[0]
    tmp_exp = (y_sent_pred[i][0] * tmp_rubric[6] + y_sent_pred[i][1] * tmp_rubric[7] + y_sent_pred[i][2] * tmp_rubric[8]) \
        / (tmp_rubric[6] + tmp_rubric[7] + tmp_rubric[8])
    tmp_org = (y_sent_pred[i][3] * tmp_rubric[10] + y_sent_pred[i][4] * tmp_rubric[11] + y_sent_pred[i][5] * tmp_rubric[12] + y_sent_pred[i][6] * tmp_rubric[13]) \
        / (tmp_rubric[10] + tmp_rubric[11] + tmp_rubric[12] + tmp_rubric[13])
    tmp_con = (y_sent_pred[i][7] * tmp_rubric[15] + y_sent_pred[i][8] * tmp_rubric[16] + y_sent_pred[i][9] * tmp_rubric[17] + y_sent_pred[i][10] * tmp_rubric[18]) \
        / (tmp_rubric[15] + tmp_rubric[16] + tmp_rubric[17] + tmp_rubric[18])
    tmp_pred_score = (
        tmp_exp * tmp_rubric[5] + tmp_org * tmp_rubric[9] + tmp_con * tmp_rubric[14]) / 10
    weighted_pred_list.append(tmp_pred_score)

    tmp_exp = (y_test[i][0] * tmp_rubric[6] + y_test[i][1] * tmp_rubric[7] + y_test[i][2] * tmp_rubric[8]) \
        / (tmp_rubric[6] + tmp_rubric[7] + tmp_rubric[8])
    tmp_org = (y_test[i][3] * tmp_rubric[10] + y_test[i][4] * tmp_rubric[11] + y_test[i][5] * tmp_rubric[12] + y_test[i][6] * tmp_rubric[13]) \
        / (tmp_rubric[10] + tmp_rubric[11] + tmp_rubric[12] + tmp_rubric[13])
    tmp_con = (y_test[i][7] * tmp_rubric[15] + y_test[i][8] * tmp_rubric[16] + y_test[i][9] * tmp_rubric[17] + y_test[i][10] * tmp_rubric[18]) \
        / (tmp_rubric[15] + tmp_rubric[16] + tmp_rubric[17] + tmp_rubric[18])
    tmp_real_score = (
        tmp_exp * tmp_rubric[5] + tmp_org * tmp_rubric[9] + tmp_con * tmp_rubric[14]) / 10
    weighted_real_list.append(tmp_real_score)

result_file = os.path.join(SAVE_DIR, 'bert_result.csv')
if not os.path.exists(result_file):
    ff = open(result_file, 'a', newline='')
    writer_ff = csv.writer(ff)
    writer_ff.writerow(['ESSAY_ID', 'REAL_SCORE', 'PRED_SCORE'])
    for i in range(len(weighted_pred_list)):
        writer_ff.writerow([test_x[i][0], weighted_real_list[i], weighted_pred_list[i]])
    ff.close()

sentence_result = cohen_kappa_score(
    np.round(weighted_real_list), np.round(weighted_pred_list), weights='quadratic')
pearson_result = np.corrcoef(
    np.round(weighted_real_list), np.round(weighted_pred_list))[0, 1]
print("Kappa Score", ": {}".format(sentence_result))
print("Pearson Correlation Coefficient", ": {}".format(pearson_result))
current = datetime.datetime.now(timezone('Asia/Seoul'))
current_time = current.replace(microsecond=0)
print("after inference : ")
print(current_time)
