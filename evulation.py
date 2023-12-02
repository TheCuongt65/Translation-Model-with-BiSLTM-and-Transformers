# import nltk.translate.bleu_score as bleu
#
# reference_translation=['The cat is on the mat.'.split(), 'There is a cat on the mat.'.split() ]
# candidate_translation_1='the the the mat on the the.'.split()
# candidate_translation_2='The cat is on the mat.'.split()
#
# print("BLEU Score candidate 1: ",bleu.sentence_bleu(reference_translation, candidate_translation_1))
# print("BLEU Score candidate 2: ",bleu.sentence_bleu(reference_translation, candidate_translation_2))

##############################

# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from nltk.util import ngrams
#
# # Giả sử chúng ta có 4 bản dịch (giả thuyết) và 4 bản dịch tham chiếu tương ứng
# hypotheses = [["this", "is", "a", "test"], ["this", "is" "test"], ["this", "is", "a", "cat"], ["this", "is"], ["this", "is"]]
# references = [["this", "is", "a", "test"], ["this", "is", "a", "test"], ["this", "is", "a", "dog"], ["this", "is", "a", "test"]]
#
# # Chuyển đổi danh sách n-gram thành tuple n-gram
# references = [[tuple(ngrams(ref[0], n)) for n in range(1, 5)] for ref in references]
# hypotheses = [tuple(ngrams(hyp, n)) for n in range(1, 5) for hyp in hypotheses]
#
# # Tính điểm BLEU cho toàn bộ bộ dữ liệu
# corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
# print("Corpus BLEU score:", corpus_bleu_score)

