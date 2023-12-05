from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

### Đánh Giá
# Các câu dịch tham chiếu
with open('data/vi_sents', 'r', encoding='utf-8') as f:
    vi_sents = f.read().split('\n')
references = [[sentence.split()] for sentence in vi_sents[0:2000]]
# print(references)
# Các câu dịch bởi model
# BiLSTM

with open('result/vi_BiLSTM.txt', 'r', encoding='utf-8') as f:
    result = f.read().split('\n')
candidatesBiLSTM = [sentence.split() for sentence in result[0:2000]]
# print(candidatesBiLSTM)

# Transformers
with open('result/vi_Trasformers.txt', 'r', encoding='utf-8') as f:
    result2 = f.read().split('\n')
candidatesTransformers = [sentence.split() for sentence in result2[0:2000]]
# print(candidatesTransformers)

# Tính toán BLEU Score cho toàn bộ tập hợp
score = corpus_bleu(references, candidatesBiLSTM, weights=(0.3,0.7,0,0))
print("Corpus BLEU Score for BiLSTM: ", score)

score = corpus_bleu(references, candidatesTransformers, weights=(0.3,0.7,0,0))
print("Corpus BLEU Score for Transformers: ", score)
