import keras
import tensorflow as tf
import keras_nlp

def tokenize(x, encode_start_end=False):
    '''
    :param x: Danh sách các câu
    :param encode_start_end: Nếu True, thêm <start> và <end> vào đầu và cuối câu.
    :return: (danh sách mã hóa của x, bộ mã hóa)
    '''
    if encode_start_end:
        x = ["startofsentence " + sentence + " endofsentence" for sentence in x] #Thêm ký đầu cuối vào câu
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x)
    tokenized_x = tokenizer.texts_to_sequences(x)
    return tokenized_x, tokenizer

def pad(x, length=None):
    '''
    :param x: Danh sách chuối cần được đệm
    :param length: Độ dài chuỗi, nếu không được chỉ định, độ dài chuỗi là độ dài của chuỗi lớn nhất trong x
    :return: mảng numpy với shape=(len(x), length)
    '''
    padded_x = keras.utils.pad_sequences(x, maxlen = length, padding = 'post', truncating = 'post')
    return padded_x

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    '''
    Hàm xử lý từ điển
    :param text_samples:
    :param vocab_size:
    :param reserved_tokens:
    :return: Danh sách từ điển
    '''
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab