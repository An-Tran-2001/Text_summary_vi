from Method.process_text import sent2vec, text2sent, sim_matrix
import numpy as np
def weight_solve(text):
    #  chuyển văn bản thành từng câu / Convert text to sentences
    sentences = text2sent(text)
    #  Chuyển câu thành vector / Convert sentences to vector
    sentences_vec = [sent2vec(i) for i in sentences]
    # xóa các giá trị nan / Remove all nan values

    sentences_vec = [i for i in sentences_vec if str(i) != 'nan']
    #  Tính ma trận độ tương tự / Calculate similarity matrix
    A = sim_matrix(sentences_vec)

    #  Tính độ tương tự trung bình của câu / Calculate the average similarity of the sentence
    avg = A.sum(axis=1) / (A.shape[0] - 1)

    #  Xây dựng trọng số cho từng câu / Build weight for each sentence
    weighted_avg = ((avg - avg.min()) / (avg.max() - avg.min()))
    #  Tính điểm cho từng câu / Calculate the score for each sentence
    score = (A.dot(weighted_avg)) / weighted_avg.sum()
    #  Sắp xếp câu theo thứ tự giảm dần / Sort sentences in descending order
    ranked_sentences = sorted(((score[i],s) for i,s in enumerate(sentences_vec)), key=lambda x: x[0], reverse=True)
    # ranked_sentences = sorted(((score[i],s) for i,s in enumerate(sentences_vec)), reverse=True)
    summary_wv = ''
    for i in range(len(ranked_sentences)):
        if ranked_sentences[i][0] > 1.05 * weighted_avg.mean():
            summary_wv += sentences[np.where(sentences_vec == ranked_sentences[i][1])[0][0]][0] + '. '
    return summary_wv