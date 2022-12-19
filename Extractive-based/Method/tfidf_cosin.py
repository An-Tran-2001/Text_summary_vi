from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Method.process_text import text2sent


def tfidf_cosin_solve(text):
    #  chuyển văn bản thành từng câu / Convert text to sentences
    sentences = text2sent(text)

    #  Tạo ma trận tf-idf / Create tf-idf matrix
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform([s[0] for s in sentences])

    #  Tính độ tương tự giữa các câu / Calculate the similarity between sentences
    sim_mat = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

    #  Tính điểm cho từng câu / Calculate the score for each sentence
    sim_scores = sim_mat.sum(axis=0)
    ranked_sentences = sorted(
        ((sim_scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    #  Chuyển các câu có điểm cao hơn 1.2 lần trọng số trung bình thành văn bản tóm tắt / Convert sentences with scores greater than 1.2 times the average weight to a summary text
    summary_tfvec = ''
    for i in range(len(ranked_sentences)):
        if ranked_sentences[i][0] > 1.05 * sim_scores.mean():
            summary_tfvec += ranked_sentences[i][1][0] + '. '
    return summary_tfvec
