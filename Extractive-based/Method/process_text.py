import numpy as np
from underthesea import sent_tokenize, word_tokenize
from Method.vector import vector
import re
#  convert sentence to vector / Hàm chuyển câu thành vector


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(vector[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    try:
        vec = v / np.sqrt((v ** 2).sum())
    except:
        pass
    return vec

#  chuyển văn bản thành từng câu / Convert text to sentences


def text2sent(text):
    # Loại bỏ các ký tự đặc biệt trừ dấu chấm
    text = re.sub(r'[^\w\s.]', '', text)
    # tách câu bằng slip
    sentences = [[sentences.lower()] for sentences in text.split('.')]
    return sentences


#  tính độ tương tự giữa 2 câu / Calculate the similarity between 2 sentences
def similarity(s1, s2):
    return np.dot(s1, s2)

#  tính độ tương tự giữa 1 câu với tất cả các câu khác / Calculate the similarity between 1 sentence and all other sentences


def sim_matrix(sentences):
    n = len(sentences)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A[i][j] = similarity(sentences[i], sentences[j])
    return A
