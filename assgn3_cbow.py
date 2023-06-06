## imports 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.manifold import TSNE
from tqdm import tqdm
from assgn3_svd import TextProcessing
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)


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

def pad_seq(history, words, target, data, pad_val=0):
    
    max_len = max(len(arr) for arr in history)

    for ind, arr in enumerate(history):
        # x = max_len - len(arr)
        for _ in range((max_len - len(arr))):
            history[ind].append(pad_val)

    data.clear()
    for ind, i in enumerate(history):
        data.append((i, words[ind], target[ind]))

    for ind, i in enumerate(data):
        data[ind] = (torch.tensor(data[ind][0]), torch.tensor(data[ind][1]), torch.tensor(data[ind][2]))

    return history, data

def neg_sampling(data, word_to_idx, corpus, doc_ind, dict_idx):
    len_corpus = len(corpus)
    neg_arr = []
    unk_word = '<UNK>'
    for _ in range(4):
        ind = random.randint(1, len_corpus -1)
        doc_ind_neg = (doc_ind + ind) % len_corpus
        
        doc_neg = corpus[doc_ind_neg]
        max_range = len(doc_neg) - 1
        rand_id = random.randint(0, max_range)

        word_neg = doc_neg[rand_id]
        try:
            ind_word_neg = word_to_idx[word_neg]
        except:
            ind_word_neg = word_to_idx[unk_word]
        neg_arr.append(ind_word_neg)
    data.append((neg_arr, dict_idx, 0))
    return data

def create_cbow(corpus, word_to_idx, data):
    window_size = 2
    len_corpus = len(corpus)
    unk_word = '<UNK>'
    for doc_ind, doc in enumerate(corpus):
        doc_size = len(doc)
        for cur_doc_idx in range(doc_size):
            l = max(0, cur_doc_idx - window_size)
            r = min(doc_size - 1, cur_doc_idx + window_size)
            w = doc[cur_doc_idx]
            try:
                dict_idx = word_to_idx[w]
            except:
                dict_idx = word_to_idx[unk_word]
            outside_words = doc[l:cur_doc_idx]+doc[cur_doc_idx+1:r+1]
            i_idx_arr = list()
            for i in outside_words:
                try:
                    i_idx = word_to_idx[i]
                except:
                    i_idx = word_to_idx[unk_word]
                i_idx_arr.append(i_idx)
            data.append((i_idx_arr, dict_idx, 1))

            # negative sampling
            data = neg_sampling(data, word_to_idx, corpus, doc_ind, dict_idx)
    
    return data


class Datasets():
    def __init__(self, vocab, corpus):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.corpus = corpus
        
        self.word_to_idx = {}
        self.window_size = 2
        self.idx_to_word = {}

        self.word_to_idx, self.corpus = create_corpus(self.corpus)

        self.word_to_idx, self.widx, self.idx_to_word = word2idx(
            self.word_to_idx)
        
        self.unk_word = '<UNK>'
        self.data = []
        # self.data_skip = list()
        self.pad_val = 0
        self.create_cbow_dataset()
        self.padding()

    def padding(self):
        # CBOW
        self.history = [sent[0] for sent in self.data]
        self.neighbours = np.array([sent[1] for sent in self.data])
        self.target = np.array([sent[2] for sent in self.data])
        
        self.history, self.data = pad_seq(self.history, self.neighbours, self.target, self.data)

    def create_cbow_dataset(self):
        self.data = create_cbow(self.corpus, self.word_to_idx, self.data)

class cbow_word2vec(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_val):
        super(cbow_word2vec, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=pad_val)
        
    def forward(self, i, o):
        in_embeds = self.in_embeddings(i).sum(dim=1)
        
        out_embeds = self.in_embeddings(o)
        
        score = torch.mm(in_embeds, torch.t(out_embeds))
        
        probs = F.logsigmoid(score)
        return probs

    def predict(self, idx):
        return self.in_embeddings(idx)


def set_cbow(data, batch_size, num_epochs, lr, w2v):
    dataload = DataLoader(data, batch_size=batch_size, num_workers=0)
    loss_func = nn.NLLLoss().to(device)
    optim = torch.optim.Adam(params=w2v.parameters(), lr=lr, weight_decay=1e-4)

    w2v = train_cbow(num_epochs, w2v, dataload, loss_func, optim)
    
    return w2v


def train_cbow(num_epochs, w2v, dataload, loss_func, optim):
    for epoch in tqdm(range(num_epochs), desc='epoch'):
        w2v.train()
        total_loss = 0
        for ind, i in enumerate(dataload):
            context, word, target = map(lambda x:x.to(device), i)

            optim.zero_grad()
            pred = w2v(context, word)
            loss = loss_func(pred, target)

            loss.backward()
            optim.step()

            total_loss += loss.item() 
        
        print("Epoch: {}/{}...".format(epoch+1, num_epochs),
              "Loss: {:.4f}...".format(total_loss/len(dataload)))

    return w2v

class CBOW_model():
    def __init__(self, vocab_size, data, pad_val=0, hidden_size=64, batch_size=512,lr=0.001, num_epochs=8, window_size=2, embedding_size=100):
        self.vocab_size = vocab_size
        self.pad_val = pad_val
        self.data = data
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.w2v = cbow_word2vec(vocab_size, embedding_size, pad_val).to(device)
        

    def train_cbow_word2vec(self):
        self.w2v = set_cbow(self.data, self.batch_size, self.num_epochs, self.lr, self.w2v)
        torch.save(self.w2v.state_dict(), 'cbow.pth')

    def nearest_words(self, word, word_to_idx):
        embeddings = self.w2v.in_embeddings.weight.data.cpu().numpy()
        tops = find_nearest_words(embeddings, word_to_idx, word)
        return tops

    def load_model(self):
        self.w2v.load_state_dict(
            torch.load(
                "cbow.pth", map_location=torch.device("cpu")
            )
        )
        return self.w2v.state_dict()


def find_nearest_words(word_embeddings, word_to_idx, word):
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
        top_indices = find_nearest_words(embeddings, word_to_idx, word)
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

    plt.savefig('tsne_cbow')
    plt.show()


if __name__ == '__main__':
    
    path = 'reviews.json'

    sentences = 45000
    textProcesser = TextProcessing(sentences)

    word_to_idx = textProcesser.word_to_idx
    idx_to_word = textProcesser.idx_to_word
    vocab = textProcesser.vocab
    
    print(vocab)
    print(len(vocab))

    cbow = Datasets(vocab, textProcesser.corpus)
    
    w2v_cbow = CBOW_model(cbow.vocab_size, cbow.data)

    # w2v_cbow.train_cbow_word2vec()

    w2v_cbow.load_model()
    w2v_cbow.nearest_words("titanic", word_to_idx)
    w2v_cbow.nearest_words("camera", word_to_idx)

    model = cbow_word2vec(cbow.vocab_size, 100, 0).to(device)
    model.load_state_dict(
        torch.load(
            "cbow.pth", map_location=torch.device("cpu")
        )
    )
    embeddings = model.in_embeddings.weight.data.numpy()

    # select five different words to plot
    words = ['good', 'music', 'happy', 'dog', 'movie']
    tsne(words, embeddings)
