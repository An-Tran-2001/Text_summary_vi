from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from Method.process_text import sent2vec, text2sent


def kmeans_w2v_skipgama_solve(text):
    #  chuyển văn bản thành từng câu / Convert text to sentences
    sentences = text2sent(text)
    #  Chuyển câu thành vector / Convert sentences to vector
    sentences_vec = [sent2vec(i) for i in sentences]
    # xóa các giá trị nan / Remove all nan values
    sentences_vec = [i for i in sentences_vec if str(i) != 'nan']

    kmeans = KMeans(n_clusters=int(len(sentences)/2),
                    random_state=0).fit(sentences_vec)

    #  Tìm ra các câu có cùng cluster / Find sentences with the same cluster
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, sentences_vec)

    #  Chuyển các câu có cùng cluster thành văn bản tóm tắt / Convert sentences with the same cluster to a summary text

    summary_KM_WV = ''
    for i in closest:
        summary_KM_WV += sentences[i][0] + '. '
    return summary_KM_WV
