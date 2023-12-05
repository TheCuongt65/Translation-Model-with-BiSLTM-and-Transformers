import inferenceBiLSTM
import time

start_time = time.time()

# Tạo kết quả
with open('data/en_sents', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

# BiLSTM
resultBiLSTM = [inferenceBiLSTM.code_sequence(sentence) for sentence in data[4000:6000]]
# print(resultBiLSTM)
result_string = '\n'.join(resultBiLSTM)
with open('result/vi_BiLSTM.txt', 'a', encoding='utf-8') as f:
    f.write(result_string)

end_time = time.time()
print(f"Thời gian chạy là: {(end_time-start_time) / 60} phút")