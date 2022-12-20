from Method.frequency import *
from Method.lex_rank import *
from Method.tfidf_cosin import *
from Method.countvec_cosin import *
from Method.kmeans_tfidf import *
from Method.kmeans_countvec import *
from Method.weight import *
from Method.process_text import text2sent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Summarize:
    def __init__(self, text):
        self.text = text
        self.summary = ''
    def fit(self):
        self.summary_f = frequency_solve(self.text)
        self.summary_l = lex_rank_solve(self.text)
        self.summary_tc = tfidf_cosin_solve(self.text)
        self.summary_cc = countvec_cosin_solve(self.text)
        self.summary_kt = kmeans_tfidf_solve(self.text)
        self.summary_kc = kmeans_countvec_solve(self.text)
        self.summary_w = weight_solve(self.text)

    def get_summary(self):
        summary_list = [self.summary_f, self.summary_l, self.summary_tc, self.summary_cc, self.summary_kt, self.summary_kc, self.summary_w]

        #  chuyển văn bản thành từng câu / Convert text to sentences
        sentences = text2sent(self.text)
        # Tách các câu trong các summeries / Split sentences in summeries
        summary_list = [s.split('. ') for s in summary_list]

        # Tạo ma trận tf-idf / Create tf-idf matrix
        vectorizer = TfidfVectorizer()
        tf_idf_matrix = vectorizer.fit_transform([s[0] for s in sentences])

        # Tính độ tương tự giữa các câu / Calculate the similarity between sentences
        sim_mat = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

        # Tính điểm cho từng câu / Calculate the score for each sentence
        sim_scores = sim_mat.sum(axis=0)

        # Tạo một list chứa các câu có điểm cao nhất / Create a list containing the sentences with the highest score
        ranked_sentences = sorted(((sim_scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        # # Lấy câu có điểm cao nhất / Get the sentence with the highest score
        # top_sentence = ranked_sentences[0][1][0]

        # lấy ra 3 câu có điểm cao nhất / Get the 3 sentences with the highest score
        top_sentence = ranked_sentences[0][1][0] + '. ' + ranked_sentences[1][1][0] + '. ' + ranked_sentences[2][1][0] + '. '

        # lấy ra các câu có điểm cao hơn 1.2 lần điểm trung bình / Get the sentences with the highest score more than 1.2 times the average score
        # top_sentence = ''
        # for i in range(len(ranked_sentences)):
        #     if ranked_sentences[i][0] > 1.2 * sim_scores.mean():
        #         top_sentence += ranked_sentences[i][1][0] + '. '

        # In các câu / Print the sentences
        return top_sentence