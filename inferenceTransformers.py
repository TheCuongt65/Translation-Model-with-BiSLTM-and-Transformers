import tensorflow as tf
from keras.models import load_model
import keras_nlp
import pickle

MAX_SEQUENCE_LENGTH = 40

transformer = load_model('result/Transformers/transformers_epoch10')
# transformer = load_model('result/Transformers/transformers_epoch1.keras')
print(transformer.summary())

with open('result/Transformers/vi_tokenizer_epoch10.pickle', 'rb') as handle:
    vi_tokenizer = pickle.load(handle)
with open('result/Transformers/eng_tokenizer_epoch10.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)


def decode_sequences(input_sentences):
    input_sentences = tf.constant([input_sentences.lower()])
    batch_size = tf.shape(input_sentences)[0]

    # Tokenize the encoder input.
    encoder_input_tokens = eng_tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def next(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache

    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = tf.fill((batch_size, 1), vi_tokenizer.token_to_id("[START]"))
    pad = tf.fill((batch_size, length - 1), vi_tokenizer.token_to_id("[PAD]"))
    prompt = tf.concat((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        end_token_id=vi_tokenizer.token_to_id("[END]"),
        index=1,  # Start sampling after start token.
    )
    generated_sentences = vi_tokenizer.detokenize(generated_tokens)
    return generated_sentences

def code_sequence(input_sentence):
    translated = decode_sequences(input_sentence)
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    return translated

if __name__ == '__main__':
    # test_eng_texts = [pair[0] for pair in test_pairs]
    test_eng_texts = ["Please put the dustpan in the broom closet"]
    print(test_eng_texts)
    for i in range(3):
        input_sentence = test_eng_texts[i]
        translated = code_sequence(input_sentence)
        print(f"** Example {i} **")
        print(input_sentence)
        print(translated)
        print()