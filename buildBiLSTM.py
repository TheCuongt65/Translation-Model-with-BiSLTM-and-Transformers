import tensorflow as tf
import keras_nlp
import keras
import processText
import numpy as np
import pickle

epochs = 10

with open('data/vi_sents', encoding='utf-8') as file:
    vi = file.read().split('\n')

with open('data/en_sents', encoding='utf-8') as file:
    en = file.read().split('\n')

#Tokenized các câu thành index sequence
eng_tokenized, eng_tokenizer = processText.tokenize(en)
vi_tokenized, vi_tokenizer = processText.tokenize(vi, encode_start_end=True)

#Padding các câu để cùng độ dài
pad_eng_tokenized = processText.pad(eng_tokenized)
pad_vi_tokenized = processText.pad(vi_tokenized)

#Kích thước từ điển
length_vocab_eng = len(eng_tokenizer.word_index)
length_vocab_fre = len(vi_tokenizer.word_index)

#Kích thước sau khi padding
print("Độ dài câu của English: ", len(pad_eng_tokenized[0]))
print("Độ dài câu của Vietnamese: ", len(pad_vi_tokenized[0]))

# 1. Encoder
encoder_inputs = keras.layers.Input(shape=(None, ), name = 'encoder_input')

embedded_seq_encoder = keras.layers.Embedding(input_dim=length_vocab_eng, #Kích thước của từ điển tức là kích thước đầu vào
                                 output_dim=300 #Kích thước nhúng.
                                 )(encoder_inputs)
encoder_bilstm = keras.layers.Bidirectional(keras.layers.LSTM(units=256,
                    activation='tanh',
                    return_sequences=False, #h_t #Có nên trả về giá trị ở mỗi bước thời gian, nếu là True thì chỉ quan tâm bước thời gian cuối cùng.
                    return_state=True,
                    name = 'bi_lstm'#c_t #Có nên trả về giá trị trạng thái ẩn cuối cùng, nếu là True trả về trạng thái ẩn cuối cùng.
                    ),
                    merge_mode='concat')

_, forward_last_hidden_encoder, forward_last_cell_encoder, backward_last_hidden_encoder, backward_last_cell_encoder = encoder_bilstm(embedded_seq_encoder)


# Cộng hai trạng thái cell cuối cùng
concatenated_last_hidden_encoder = keras.layers.Concatenate(name = 'concatenated_last_hidden_encoder')([forward_last_hidden_encoder, backward_last_hidden_encoder])

# Nối hai trạng thái cell cuối cùng
concatenated_last_cell_encoder = keras.layers.Concatenate(name = 'concatenated_last_cell_encoder')([forward_last_cell_encoder, backward_last_cell_encoder])


#2. Decoder
input_seq_decoder = keras.layers.Input(shape = (None, 1), name='input_encode')

decoder_lstm = keras.layers.LSTM(units=512,
                    activation='tanh',
                    return_sequences=True,
                    return_state=True,
                    name = 'decoder_lstm')

all_hidden_decoder,_,_ = decoder_lstm(input_seq_decoder,
                                      initial_state=[concatenated_last_hidden_encoder,
                                                     concatenated_last_cell_encoder] #Trạng thái ẩn và trạng thái cell cuối cùng ở encoder
                                      )

decoder_dense = keras.layers.Dense(length_vocab_fre, #Kích thước của từ điển France
                      activation="softmax",
                      name = 'decoder_dense')
logits = decoder_dense(all_hidden_decoder)

#3. Define model
seq2seqModel = keras.models.Model(inputs = [encoder_inputs, input_seq_decoder],
                                  outputs = logits)
seq2seqModel.compile(loss = 'sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

# Xem thông tin model
print(seq2seqModel.summary())


# Chuẩn bị data
## Input
english_input = pad_eng_tokenized[:, :, np.newaxis]
decoder_vi_input = pad_vi_tokenized[:, :-1,np.newaxis]

## Target
decoder_vi_target = pad_vi_tokenized[:, 1:, np.newaxis]


## Train model
seq2seqModel.fit([english_input, decoder_vi_input],
                 decoder_vi_target,
                 batch_size=200,
                 epochs=epochs,
                 validation_split=0.2)

## Tạo bộ mã hóa
last_states_encoder = [concatenated_last_hidden_encoder, concatenated_last_cell_encoder]
inference_encoder_model = keras.models.Model(inputs = encoder_inputs,
                                             outputs = last_states_encoder)

## Tạo bộ giải mã
initial_states_decoder = [keras.layers.Input(shape = (512,)), keras.layers.Input(shape = (512,))]
all_hidden_decoder, last_hidden_decoder, last_cell_decoder = decoder_lstm(input_seq_decoder,
                                                                          initial_state=initial_states_decoder)
logic = decoder_dense(all_hidden_decoder)

inference_decoder_model = keras.models.Model(inputs = [input_seq_decoder] + initial_states_decoder,
                                             outputs = [logic, last_hidden_decoder, last_cell_decoder])

# Lưu lại mã hóa - giải mã và tokenizer
## Lưu lại bộ mã hóa và giải mã
seq2seqModel.save(f'result/BiLSTM/seq2seq_epoch{epochs}.keras')
inference_encoder_model.save(f'result/BiLSTM/model_encoder_BiLSTM_epoch{epochs}.keras')
inference_decoder_model.save(f'result/BiLSTM/model_decoder_BiLSTM_epoch{epochs}.keras')

## Lưu lại tokenizer
with open(f'result/BiLSTM/tokeizerBiLSTM_epoch{epochs}.pickle', 'wb') as f:
    pickle.dump(eng_tokenizer, f)
    pickle.dump(vi_tokenizer, f)

# Tải lại tokenizer
# with open('result/tokeizerBiLSTM.pickle', 'rb') as f:
#     eng_tokenizer = pickle.load(f)
#     vi_tokenizer = pickle.load(f)
'''
Epoch 2:
    Acc :
        Train 0.9082 
        Val 0.9150 
    Loss 0.5105
    Time: 12.8p
    Bleu:
Epoch 5:
    Acc :
        Train 0.9379
        Val 0.9376
    Loss 0.2954
    Time: 31.85p
    Bleu:
    
Epoch 10:
    Acc :
        Train 0.9565
        Val 0.9484
    Loss: 0.1836 
    Time: 57.25p
    Bleu: 
        
'''