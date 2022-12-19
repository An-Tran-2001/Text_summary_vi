import os
stop_words = open(os.getcwd()+'/Data/stop_word/vietnamese-stopwords.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.split()