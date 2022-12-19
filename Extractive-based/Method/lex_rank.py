from lexrank import LexRank
# import stop_words.stop_words as stop_words
import os
stop_words = open(os.getcwd()+'/Data/stop_word/vietnamese-stopwords.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.split()

#  Tóm tắt văn bản với phương pháp LexRank


def lex_rank_solve(text):
    text = text.split('.')
    lxr = LexRank(text, stopwords=set(stop_words))
    summary_LexRank = ''
    for i in lxr.get_summary(text, summary_size=3, threshold=None):
        summary_LexRank += i + '. '
    return summary_LexRank
