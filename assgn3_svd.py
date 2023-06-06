# imports
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from warnings import filterwarnings
import gensim.downloader as api

filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)


def clean_text(text):
    text = '\n'.join(text)

    text = re.sub(r'\n+', r'\.', text)

    text = re.sub("Mr\s*\.", "Mr", text)
    text = re.sub("Dr\s*\.", "Dr", text)
    text = re.sub("Mrs\s*\.", "Mrs", text)

    text = text.lower()

    return text


def replace_abbreviations(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r'(\w+)\'ve', r'\1 have', text)
    text = re.sub(r'(\w+)\'re', r'\1 are', text)
    text = re.sub(r'(\w+)\'t', r'\1 not', text)
    text = re.sub(r'(\w+)\'s', r'\1 has', text)
    text = re.sub(r'(\w+)\'ll', r'\1 will', text)

    return text


class Tokeniser():
    def __init__(self):
        pass

    def tokenise(self):
        self.text = clean_text(self.text)

    def modify_text(self, text):
        self.text = text
        self.tokenise()

        self.text = re.sub(r'[^\w+^\.^\?^\!\s]', r'', self.text)

        self.text = replace_abbreviations(self.text)

        self.text = re.sub(' +', ' ', self.text)

        self.text = re.split('\w*\.\w* | \w*\?\w* | \w*\!\w*', self.text)

        for i in self.text:
            if i == '':
                self.text.remove(i)

        return self.text

# preprocessing text and creating vocab


def process_corpus(text, corpus, length):
    check = 0
    for sentences in text:
        sentences = sentences.split('.')
        for word in sentences:
            corpus.add(word)
        if len(corpus) > length:
            check += 1
            break

    return check, corpus


def word2idx(word_to_idx):
    word_to_idx = dict(
        sorted(word_to_idx.items(), key=lambda x: x[0]))
    widx = word_to_idx.copy()
    word_to_idx = {word: i for i, word in enumerate(
        list(word_to_idx.keys()))}
    word_to_idx['<pad>'] = 0
    idx_to_word = {i: word for i,
                   word in enumerate(list(widx.keys()))}
    idx_to_word[0] = '<pad>'

    return word_to_idx, widx, idx_to_word


def create_corpus(corpus):
    word_to_idx = defaultdict(int)

    for doc in corpus:
        for word in doc:
            word_to_idx[word] += 1
    for k in list(word_to_idx.keys()):
        if ('!' in k):
            del word_to_idx[k]
        elif word_to_idx[k] == 1:
            word_to_idx['<UNK>'] += 1
            del word_to_idx[k]

    return word_to_idx, corpus


def clean_corpus(corpus):
    for i in corpus:
        if len(i.split()) <= 1:
            corpus.remove(i)
    return corpus


class TextProcessing():
    def __init__(self, sentences):
        self.path = 'reviews.json'
        self.sentences = sentences
        self.pad_token = 0
        self.indenture()

    def read_data(self):
        check = 0
        for block in self.chunks:
            text = block['reviewText']

            check, self.corpus = process_corpus(
                text, self.corpus, self.sentences)

            if check:
                self.corpus = list(self.corpus)
                break

        self.corpus = clean_corpus(self.corpus)

        self.corpus = self.tokeniser.modify_text(self.corpus)
        self.corpus = [["SOS"]+[word for word in sentence.split()]+["END"]
                       for sentence in self.corpus]

    def build_vocab(self):
        self.vocab = []
        self.vocab = sorted(
            list(set([word for doc in self.corpus for word in doc])))
        self.word_to_idx = {}

        self.word_to_idx, self.corpus = create_corpus(self.corpus)

        self.word_to_idx, self.widx, self.idx_to_word = word2idx(
            self.word_to_idx)

        self.vocab = list(self.word_to_idx.keys())
        self.vocab_size = len(self.vocab)

    def indenture(self):
        self.vocab = set()
        self.corpus = set()
        self.tokeniser = Tokeniser()
        self.chunks = pd.read_json(self.path, lines=True, chunksize=20000)
        self.read_data()
        self.build_vocab()

# performing method #1 svd


def make_cooccurence_matrix(corpus, vocab_size, window_size, word_to_idx):
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for doc in corpus:
        doc_size = len(doc)
        for cur_doc_idx in range(doc_size):
            w = doc[cur_doc_idx]
            try:
                dict_idx = word_to_idx[w]
            except:
                dict_idx = word_to_idx["<UNK>"]
            fringe_words = doc[max(0, cur_doc_idx - window_size):cur_doc_idx] + \
                doc[cur_doc_idx+1:min(doc_size - 1, cur_doc_idx + window_size)]
            for i in fringe_words:
                try:
                    i_idx = word_to_idx[i]
                except:
                    i_idx = word_to_idx["<UNK>"]
                matrix[i_idx, dict_idx] += 1

    return matrix


class SVD():
    def __init__(self, corpus, dict, vocab, vocab_size):
        self.corpus = corpus
        self.word_to_idx = dict
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.window_size = 4
        self.embedding_dim = 50
        self.matrix = np.zeros(
            (self.vocab_size, self.vocab_size), dtype=np.int32)
        self.unk_word = '<UNK>'

        self.build_matrix()
        self.cal_svd()

    def build_matrix(self):
        print('Creating the Co-Occurrence Matrix')

        self.matrix = make_cooccurence_matrix(
            self.corpus, self.vocab_size, self.window_size, self.word_to_idx)

        print('Co-Occurrence Matrix Created')

    def cal_svd(self):
        print('Doing the Singular Value Decomposition')
        self.svd = TruncatedSVD(n_components=self.embedding_dim, n_iter=10)
        self.word_embeddings = self.svd.fit_transform(self.matrix)
        print('SVD Done')

# def plot(word_embeddings, word_to_idx, words):
#     for word in words:
#         try:
#             idx = word_to_idx[word]
#         except:
#             idx = word_to_idx['<UNK>']
#         x, y = word_embeddings[idx]
#         plt.scatter(x, y, marker='o', color='blue')
#         plt.text(x, y, word, fontsize=9)
#     plt.savefig('svd.png')
#     plt.show()

def find_nearest_words(word_embeddings, word_to_idx, idx_to_word, word):
    try:
        idx = word_to_idx[word]
    except:
        idx = word_to_idx['<UNK>']

    word_embed = word_embeddings[idx].reshape(1, -1)

    similarity_scores = cosine_similarity(word_embed, word_embeddings)

    tops = [idx_to_word[i] for i in similarity_scores.argsort()[0][::-1][:11]]

    print("SVD: ", word)
    print(tops)


def get_word_vectors(word_embeddings, word_to_idx, words):
    word_vectors = list()
    for word in words:
        try:
            idx = word_to_idx[word]
        except:
            idx = word_to_idx['<UNK>']
        
        word_vectors.append(word_embeddings[idx])

    word_vectors = np.array(word_vectors)
    return word_vectors



def nearest_words(word_embeddings, word_to_idx, word):
    try:
        idx = word_to_idx[word]
    except:
        idx = word_to_idx['<UNK>']
    similarity_scores = cosine_similarity(word_embeddings[idx].reshape(1, -1), word_embeddings)
    top_indices = similarity_scores.argsort()[0][::-1][:11]

    tops = [idx_to_word[i] for i in top_indices]

    print(tops)

    return top_indices


def tsne(words, embeddings):
    word_indices = [word_to_idx[word] for word in words]
    top_k = 10
    similar_word_indices = []
    for word in words:
        top_indices = nearest_words(embeddings, word_to_idx, word)
        similar_word_indices.append(top_indices)

    selected_words = []

    # iterate over each list of similar word indices in similar_word_indices
    for similar_word_indices_list in similar_word_indices:
        for similar_word_idx in similar_word_indices_list:
            word = idx_to_word[similar_word_idx]
            selected_words.append(word)

    similar_vectors = []

    # iterate over each list of similar word indices in similar_word_indices
    for similar_word_indices_list in similar_word_indices:
        for similar_word_idx in similar_word_indices_list:
            embedding_vector = embeddings[similar_word_idx]
            similar_vectors.append(embedding_vector)
    similar_vectors = np.vstack(similar_vectors)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_vectors = tsne.fit_transform(similar_vectors)
    colors = ['r', 'g', 'b', 'c', 'm']

    c = -1
    fig, ax = plt.subplots(figsize=(25,25))
    for c, color in enumerate(colors):
        for ind in range(c*11, (c+1)*11):
            x_val, y_val = tsne_vectors[ind]
            ax.scatter(x_val, y_val, color=color)
            ax.annotate(selected_words[ind], (x_val+0.005, y_val+0.005), fontsize=10)

    plt.title('t-SNE Visualization')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')

    plt.savefig('tsne_svd')
    plt.show()


def pretrained_res():
    # Download the pre-trained embeddings
    pretrained_model = api.load("word2vec-google-news-300")

    # Get the top 10 closest words to "titanic" in the pre-trained embeddings
    word = "titanic"
    pretrained_closest_words = pretrained_model.most_similar(word, topn=10)

    print("Word2Vec: {}".format(word))
    # print(pretrained_closest_words)
    # Print only the word
    for word, score in pretrained_closest_words:
        print(word)

    word = "camera"
    pretrained_closest_words = pretrained_model.most_similar(word, topn=10)

    print("Word2Vec: {}".format(word))
    # print(pretrained_closest_words)
    for word, score in pretrained_closest_words:
        print(word)

def scale(word_embeddings):
    scaler = StandardScaler()
    word_embeddings = scaler.fit_transform(word_embeddings)
    return word_embeddings

if __name__ == '__main__':

    path = 'reviews.json'

    sentences = 45000
    textProcesser = TextProcessing(sentences)
    print(textProcesser.vocab)
    print(len(textProcesser.vocab))
    
    word_to_idx = textProcesser.word_to_idx
    idx_to_word = textProcesser.idx_to_word

    svd = SVD(textProcesser.corpus, textProcesser.word_to_idx,
              textProcesser.vocab, textProcesser.vocab_size)

    f = open('svd_1.npy', 'wb')
    np.save(f, svd.word_embeddings, allow_pickle=True)
    f.close()

    word_embeddings = svd.word_embeddings

    word_embeddings = scale(word_embeddings)

    words = ["funny", "hilarious", "good", "bad",
             "king", "beggar", "titanic", "giant"]
    # plot(word_embeddings, word_to_idx, words)

    pretrained_res()

    word = "titanic"
    find_nearest_words(
        svd.word_embeddings, word_to_idx, idx_to_word, word)
    
    word = "camera"
    find_nearest_words(
        svd.word_embeddings, word_to_idx, idx_to_word, word)

    words = ['good', 'music', 'happy', 'dog', 'movie']
    tsne(words, word_embeddings)
