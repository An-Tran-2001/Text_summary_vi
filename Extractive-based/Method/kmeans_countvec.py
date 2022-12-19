from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import CountVectorizer
from Method.process_text import text2sent


def kmeans_countvec_solve(text):
    #  chuyển văn bản thành từng câu / Convert text to sentences
    sentences = text2sent(text)
    
    #  Tạo ma trận tf-idf / Create tf-idf matrix
    vectorizer_C = CountVectorizer()
    tf_idf_matrix_C = vectorizer_C.fit_transform([s[0] for s in sentences])

    #  Tạo model KMeans / Create KMeans model
    kmeans = KMeans(n_clusters=int(len(sentences)/2),
                    random_state=0).fit(tf_idf_matrix_C)

    #  Tìm ra các câu có cùng cluster / Find sentences with the same cluster
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, tf_idf_matrix_C)

    #  Chuyển các câu có cùng cluster thành văn bản tóm tắt / Convert sentences with the same cluster to a summary text
    summary_KM_C = ''
    for i in closest:
        summary_KM_C += sentences[i][0] + '. '
    return summary_KM_C
