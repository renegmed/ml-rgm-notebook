import tensorflow as tf
import argparse
import os
import numpy as np
import json

def prep_data():
    data_url = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    
    dataset_text = open(data_url, 'rb').read().decode(encoding='utf-8')
    
    # obtain the unique characters in the dataset and print out their length
    vocab = sorted(set(dataset_text))
    
    ## Step #3: MAP TEXT TO NUMBERS
    # In this step, we are going to map the characters to numbers
    
    # Creating a mapping from unique characters to indices
    char2idx = {char: index for index, char in enumerate(vocab)}
#     {'\n': 0,
#      ' ': 1,
#      '!': 2,
#      '$': 3,
#      '&': 4,
#      "'": 5,
#      ',': 6,
#      '-': 7,
#      '.': 8,
#      '3': 9,

    idx2char = np.array(vocab)
    # array(['\n',' ','!','$','&',.....])

    # Convert dataset from 'characters' to 'integers'
    text_as_int = np.array([char2idx[char] for char in dataset_text])
    # array([18,47,56,...,45,8,0])
    
    # Show how the first 13 characters from the text are mapped to integers
    # print ('{} ---- characters mapped to int ---- > {}'.format(repr(dataset_text[:13]), text_as_int[:13]))
    # 'First Citizen' --- characters mapped to int ---- > [18 47 56 57 58  1 15 47 58 47 64 43 52]

    # Step #4: CREATE TRAINING SAMPLES AND BATCHES
    # Divide the dataset into a sequence of characters with seq_length
    # The output will be the same as the input but shifted by one character
    # Example: if out text is "Hello" and seq_len = 4
    #   Input: "Hell"
    #   Output: "ello"
    
    # Calculate the number of examples per epoch assuming a sequence length of 100 characters
    seq_length = 100
    examples_per_epoch = len(dataset_text)//seq_length
    # 11153
    
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    # Try a different number, let's say 200
#     for i in char_dataset.take(200):
#         print(idx2char[i.numpy()])
    # A
    # l
    # l
    # .
    # .
    
    # The batch method let us easily convert these individual characters to sequences of the desired size.
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
#     for item in sequences.take(10):
#         print(repr(''.join(idx2char[item.numpy()])))
    # 'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen .....'
    # 'are all resolved rather to die than to famish?\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst.......'
    #'now Caius Marcius is chief enemy to the people....'
    
    
    # For each sequence, duplicate and shift it to form the input and target text
    # by using the 'map' method to apply a simple future
    
    dataset = sequences.map(split_input_target)
    
    # Print the first examples input and target values:
#     for input_example, target_example in dataset.take(10):
#         print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#         print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))
        
#     Input data: 'First Citizen:\nBefore we proceed....'
#     Target data: 'irst Citizen:\nBefore we proceed....'
#     Input data: 'are all resolved rather to die...' 
#     Target data: 're all resolved rather to die...'
        
    # Shuffle the dataset and it into batches
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset, vocab, BATCH_SIZE


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
    
def build_and_train_model(dataset, vocab, batch_size):
    # Step #5: BUILD AND TRAIN THE MODEL
    # Use tf.keras.Sequential to define the model. Three layers are used:
    #   tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map
    #                              the numbers of each character to a vector with embedding_dim dimensions
    #   tf.keras.layers.LSTM
    #   tf.keras.layers.Dense: The output layer, with vocab_size outputs
    
    # len(vocab)
    # 65
    
    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    
    # The embedding dimension
    embedding_dim = 256
    
    # Number of RNN units
    rnn_units = 1024 # i.e. LSTM units
    
    model = build_model(
        vocab_size = len(vocab),
        embedding_dim = embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size
    )
    # Train the model
    
    # At this point the problem can be treated as a standard classification problem.
    # Given the previous RNN state, and the input this time step, predict the 
    # class of the next character.
    
    model.compile(optimizers='adam', loss=loss)
    
    # Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training:
    
#     # Directory where the checkpoints will be saved
#     checkpoint_dir = './training_checkpoints'
        
#     # Name of the checkpoint files
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
#     checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_prefix,
#         save_weights_only=True
#     )
    
    # Execute training
    
    EPOCHS=2
    
    model.fit(dataset, epochs=EPOCHS) #, callbacks=[checkpoint_callback])
    
    return model
    
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                 batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, 
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())
    


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
        
    dataset, vocab, batch_size = prep_data()
    model = build_and_train_model(dataset, vocab, batch_size)
    
    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        model.save(os.path.join(args.sm_model_dir, '000000001'), 'rnn_model.h5')
    