import random
import processText
import keras_nlp
import keras
import tensorflow as tf
import pickle

BATCH_SIZE = 64
EPOCHS = 10  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 3614
SPA_VOCAB_SIZE = 4989

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


with open('data/vi_sents', encoding='utf-8') as file:
    vi = file.read().split('\n')

with open('data/en_sents', encoding='utf-8') as file:
    en = file.read().split('\n')

text_pairs = list(zip([item.lower() for item in en], vi))

# Chia train test val
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = processText.train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

vi_samples = [text_pair[1] for text_pair in train_pairs]
vi_vocab = processText.train_word_piece(vi_samples, SPA_VOCAB_SIZE, reserved_tokens)

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=eng_vocab,
                                                        lowercase=False)
vi_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vi_vocab,
                                                        lowercase=False)


def preprocess_batch(eng, vi):
    batch_size = tf.shape(vi)[0]

    eng = eng_tokenizer(eng)
    vi = vi_tokenizer(vi)

    # Thêm [PAD] vào 'eng' (ngôn ngữ cần dịch) để đủ độ dài MAX_SEQUENCE_LENGTH
    eng_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]")
    )

    eng = eng_start_end_packer(eng)

    # Thêm [PAD] [START] [PAD] vào và đệm cho câu
    spa_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=vi_tokenizer.token_to_id("[START]"),
        end_value=vi_tokenizer.token_to_id("[END]"),
        pad_value=vi_tokenizer.token_to_id("[PAD]")
    )

    vi = spa_start_end_packer(vi)

    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": vi[:, :-1],
        },
        vi[:, 1:]  # output, target
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)

    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


# Build model
# Encoder
encoder_inputs = keras.Input(shape=(None, ),
                             dtype="int64",
                             name = "encoder_inputs")

embedded_seq_encoder = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=ENG_VOCAB_SIZE,
                                                                  sequence_length=MAX_SEQUENCE_LENGTH,
                                                                  embedding_dim=EMBED_DIM,
                                                                  mask_zero=True
                                                                  )(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM,
                                                      num_heads=NUM_HEADS)(inputs = embedded_seq_encoder)

encoder = keras.Model(encoder_inputs, encoder_outputs) # out: shape = (batch_size, sequence_length, embedding_dim)

# Decoder

decoder_inputs = keras.Input(shape=(None, ),
                             dtype="int64",
                             name = "decoder_inputs")

encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM),
                                 name = "decoder_state_inputs")

embedded_seq_decoder = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=SPA_VOCAB_SIZE,
                                                                  sequence_length=MAX_SEQUENCE_LENGTH,
                                                                  embedding_dim=EMBED_DIM,
                                                                  mask_zero=True
                                                                  )(decoder_inputs)
decoder_transfomers = keras_nlp.layers.TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM,
                                                          num_heads=NUM_HEADS
                                                          )(decoder_sequence = embedded_seq_decoder,
                                                            encoder_sequence = encoded_seq_inputs)

decoder_transfomers = keras.layers.Dropout(0.5)(decoder_transfomers)

decoder_output = keras.layers.Dense(SPA_VOCAB_SIZE,
                                    activation='softmax')(decoder_transfomers) #out: shape(batch_size, max_sequence, spa_vocab_size)

decoder = keras.Model(
    inputs = [decoder_inputs,encoded_seq_inputs],
    outputs = decoder_output
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(inputs = [encoder_inputs, decoder_inputs],
                          outputs = decoder_outputs,
                          name='transformer')

print(transformer.summary())

## Train
transformer.compile(optimizer='rmsprop',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

## Lưu lại tokenizer
with open(f'result/Transformers/eng_tokenizer_epoch{EPOCHS}.pickle', 'wb') as handle:
    pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'result/Transformers/vi_tokenizer_epoch{EPOCHS}.pickle', 'wb') as handle:
    pickle.dump(vi_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Lưu lại model
transformer.save(f'result/Transformers/transformers_epoch{EPOCHS}')


###
'''
Epoch 2: 
    Time 284 + 279
    Acc
        Train 0.7354
        Val 0.7615
    Loss
        Train 0.4138
        Val 0.3668
    BLEU
        
Epoch 5: 
    Time 219 + 212 + 213+ 212 + 212
    Acc
        Train 0.7934
        Val 0.7894 
    Loss
        Train 0.3285
        Val 0.3315 
    BLEU
Epoch 10:
    Time 219 + 212 + 213+ 212 + 212 + 222 + 214 + 213 + 213 + 215
    Acc
        Train 0.8271
        Val 0.8033
    Loss
        Train 0.2765
        Val 0.3163
    BLEU
'''
