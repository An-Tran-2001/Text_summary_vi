from stop_words import stop_words
from underthesea import word_tokenize, sent_tokenize
#  Tóm tắt văn bản với phương pháp tần suất
def solve(text):
    words = word_tokenize(text) # Tách từ
    freqTable = {}# Tạo bảng tần suất(là 1 cái dict)
    for word in words:# Đếm số lần xuất hiện của từ trong văn bản
        word = word.lower()# Chuyển về chữ thường
        if word in stop_words:# Nếu từ là stop word thì bỏ qua
            continue# Bỏ qua các từ không có nghĩa
        if word in freqTable:# Nếu từ đã có trong bảng tần suất thì tăng số lần xuất hiện lên 1
            freqTable[word] += 1
        else :# Nếu từ chưa có trong bảng tần suất thì thêm vào bảng tần suất
           freqTable[word] = 1

    sentences = sent_tokenize(text)# Tách câu
    sentenceValue = {}# Tạo bảng giá trị câu(là 1 cái dict)
    for sentence in sentences:# Tính giá trị của câu
        for word, freq in freqTable.items():# Duyệt từng từ trong bảng tần suất
            if word in sentence.lower():# Nếu từ có trong câu thì cộng giá trị của từ vào giá trị của câu
                if sentence in sentenceValue:# Nếu câu đã có trong bảng giá trị câu thì cộng giá trị của từ vào giá trị của câu
                    sentenceValue[sentence] += freq
                else :# Nếu câu chưa có trong bảng giá trị câu thì thêm vào bảng giá trị câu
                    sentenceValue[sentence] = freq
    
    sumValues = 0# Tổng giá trị của tất cả các câu
    for sentence in sentenceValue:# Tính tổng giá trị của tất cả các câu
        sumValues += sentenceValue[sentence]# Cộng giá trị của từng câu vào tổng giá trị của tất cả các câu
    average = int(sumValues / len(sentenceValue))# Tính giá trị trung bình của tất cả các câu

    summary = ''# Chuỗi tóm tắt
    for sentence in sentences:# Tạo tóm tắt
        if (sentence in sentenceValue) and(sentenceValue[sentence] > (1.2 * average)):# Nếu giá trị của câu lớn hơn 1.2 lần giá trị trung bình  của tất cả các câu thì thêm vào tóm tắt với  
            summary += "" + sentence# Thêm câu vào tóm tắt
    return summary# Trả về tóm tắt