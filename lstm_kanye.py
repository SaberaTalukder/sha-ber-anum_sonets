

import tensorflow as tf

import numpy as np
import os
import re
import urllib3

path_to_file = 'data/shakespeare.txt'
context_length = 200


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def get_training():
    # load text
    raw_text = load_doc('data/shakespeare.txt')
    #print(raw_text[:200])

    temp = re.split('\s+\d+\s',raw_text)
    
    target_url = "https://pastebin.com/raw/9S5u08EU" # Kanye lyrics
    http = urllib3.PoolManager()
    response = http.request('GET', target_url)
    lyrics = response.data.decode('utf-8')
    temp_lyrics =  re.split('\n\n',lyrics)

    text = temp+ temp+temp+temp + temp_lyrics

    reconcatenated = ''.join(text) 

    vocab = sorted(set(reconcatenated))
    #print(vocab)
    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)


    #chunks = [[] for sonnet in temp]
    chunks = [sonnet[i:i+context_length] for sonnet in text  for i in range(len(sonnet)-context_length)]
    #print(list(chunks[:10][:]))
    #text_as_int = np.array([char2idx[c] for c in text])
    chunks_as_int =[[char2idx[c] for c in chunk] for chunk in chunks]

    sequences = tf.data.Dataset.from_tensor_slices(chunks_as_int)

    dataset = sequences.map(split_input_target)

    return dataset, vocab, char2idx, idx2char


dataset, vocab, char2idx, idx2char = get_training()

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#dataset = dataset.shuffle(BUFFER_SIZE)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256
#embedding_dim = 32

# Number of RNN units
lstm_units = 200

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def build_model(vocab_size, embedding_dim, lstm_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(lstm_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  model.compile(loss=loss, optimizer='adam')
  return model

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints_kanye'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


def train_model():
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)



    model = build_model(
        vocab_size = len(vocab),
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        batch_size=BATCH_SIZE)

    print(dataset.take(1))


    model.fit(dataset, epochs=25, verbose=2, callbacks=[checkpoint_callback])
    return model


def generate_model():
    model = build_model(vocab_size, embedding_dim, lstm_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))
    print(model.summary())
    return model

def generate_text(model, start_string, length=1000, temperature=1.0):
  # Evaluation step (generating text using the learned model)

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  for i in range(length):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

#model = train_model()
model = generate_model()
#train_model()

print(generate_text(model, start_string=u"shall i compare thee to a summer's day?\n"))
print('')
print('temperature: 0.1')
print(generate_text(model, start_string=u"shall i compare thee to a summer's day?\n",temperature=0.1))
print('')
print('temperature: 0.25')
print(generate_text(model, start_string=u"shall i compare thee to a summer's day?\n",temperature=0.25))
print('')
print('temperature: 0.75')
print(generate_text(model, start_string=u"shall i compare thee to a summer's day?\n",temperature=0.75))
print('')
print('temperature: 1.5')
print(generate_text(model, start_string=u"shall i compare thee to a summer's day?\n",temperature=1.5))



