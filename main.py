import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add TensorFlow models directory to the path
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization

# Load Quora Insincere Questions Dataset
df = pd.read_csv('train.csv.zip', compression='zip', low_memory=False)

# Print dataset shape and last 20 rows
# print(f"Dataset shape: {df.shape}")
# print("Last 20 rows of the dataset:")
# print(df.tail(20))

# create tf.data.Datasets for training and evaluation

train_df, remaining = train_test_split(df, random_state=42, train_size=0.0075, stratify=df.target.values)

valid_df, _ = train_test_split(remaining, random_state=42, train_size=0.00075, stratify=remaining.target.values)

train_df.shape, valid_df.shape

with tf.device('/cpu:0'):
  train_data = tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values))
  valid_data = tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values))

  for text, label in train_data.take(1):
    print(text)
    print(label)

# download pre trained bert model from tensorflow hub
label_list = [0, 1] # Label categories
max_seq_length = 128 # maximum length of (token) input sequences
train_batch_size = 32

# Get BERT layer and tokenizer:
# More details here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
tokenizer.wordpiece_tokenizer.tokenize('hi, how are you doing?')
tokenizer.convert_tokens_to_ids(tokenizer.wordpiece_tokenizer.tokenize('hi, how are you doing?'))