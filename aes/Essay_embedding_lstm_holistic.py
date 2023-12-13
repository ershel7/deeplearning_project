import gensim
import os
import pandas as pd
import csv
from konlpy.tag import Okt
from tqdm import tqdm  # tqdm 추가

okt = Okt()
ko_model = gensim.models.Word2Vec.load('./data/ko/ko.bin')

# read_raw_data
DATASET_DIR = './data/'
SAVE_DIR = './'
X = pd.read_csv(os.path.join(
    DATASET_DIR, 'dataset.csv'), encoding='utf-8')
X.head()

# essay_data_first_fix
essay_data_origin = X['ESSAY_CONTENT']
essay_data = []

# one type
for i, data in enumerate(essay_data_origin):
    new = data.replace('<span>', '').replace(
        '</span>', '').replace('\n', '').replace('\t', '')
    essay_data.append(new)


def essay_to_sentences(essay_v):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    raw_sentences = essay_v.split('#@문장구분#')
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences = essay_to_wordlist2(raw_sentence)
            if tokenized_sentences != []:
                sentences.append(tokenized_sentences)
    return sentences


def essay_to_wordlist2(essay_v):
    """Remove the tagged labels and word tokenize the sentence."""
    token = okt.morphs(essay_v)
    return token


# tokenized essays
essays = []
for ix, essay in tqdm(enumerate(essay_data), desc="Tokenizing Essays", total=len(essay_data)):
    sentences = essay_to_sentences(essay)
    essays.append(sentences)

# # embedding & make file
ff = open(os.path.join(
    DATASET_DIR, 'embedded_features_word2vec_holistic.csv'), 'a', newline='')
writer_ff = csv.writer(ff)

sent_max_len = 50
for ix in tqdm(range(len(essays)), desc="에세이 처리 중"):
        for jx in tqdm(range(len(essays[ix])), desc="문장 처리 중", leave=False):
            sent_embedded = []
            for w in essays[ix][jx]:
                try:
                    sent_embedded.append(ko_model[w])
                except KeyError:
                    list = [0 for i in range(200)]
                    sent_embedded.append(list)
            avg_sent_embedded = []
            for kx in range(len(sent_embedded[0])):
                avg_element = 0.0
                for lx in range(len(sent_embedded)):
                    avg_element += sent_embedded[lx][kx]
                avg_element = avg_element / len(sent_embedded)
                avg_sent_embedded.append(avg_element)
            writer_ff.writerow(avg_sent_embedded)
ff.close()
