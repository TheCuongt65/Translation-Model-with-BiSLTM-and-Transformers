import numpy as np
from keras.models import load_model
import pickle
import processText

inference_encoder_model = load_model('result/BiLSTM/model_encoder_BiLSTM_epoch10.keras')
inference_decoder_model = load_model('result/BiLSTM/model_decoder_BiLSTM_epoch10.keras')

with open('result/BiLSTM/tokeizerBiLSTM_epoch10.pickle', 'rb') as f:
    eng_tokenizer = pickle.load(f)
    vi_tokenizer = pickle.load(f)

target_id_to_word = vi_tokenizer.index_word
def code_sequence(input_seq):
    seq_tokeized = eng_tokenizer.texts_to_sequences(input_seq)
    pad_eng_tokenized = processText.pad(seq_tokeized, length=63)
    english_input = pad_eng_tokenized[:, :, np.newaxis]
    decoder_input = inference_encoder_model.predict(english_input)

    prev_word = np.zeros(shape=(1, 1, 1))
    prev_word[0, 0, 0] = vi_tokenizer.word_index['startofsentence']

    flat = True
    translation = []
    while flat:
        # Dự đoán từ tiếp theo
        logic, last_h, last_c = inference_decoder_model.predict([prev_word] + decoder_input)

        # print(logic.shape)

        # Cập nhật từ tiếp theo vào bản dịch
        predicted_id = np.argmax(logic[0, 0, :])
        if predicted_id in target_id_to_word:
            predicted_word = target_id_to_word[predicted_id]
        else:
            predicted_word = "<UNK>"

        translation.append(predicted_word)

        # Kiểm tra điều kiện dừng
        if (predicted_word == 'endofsentence' or len(translation) > 50): #50 là decoder_french_target.shape[1]
              print(len(translation))
              flat = False

        # Cập nhật từ cho Cell tiếp theo
        prev_word[0, 0, 0] = predicted_id
        decoder_input = [last_h, last_c]
    return " ".join(translation).replace('endofsentence', '')


if __name__ == '__main__':
    # text = ["Are you going to come tomorrow?",
    #         "Please put the dustpan in the broom closet",
    #         "Friendship consists of mutual understanding"]
    print("I am a student.")
    print(code_sequence(["I am a student"]))