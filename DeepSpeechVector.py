#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
import os
import scipy

import numpy as np
import pandas
import tensorflow as tf
from six.moves import range
from tensorflow.python.keras.utils.conv_utils import conv_output_shape

from util.config import Config, initialize_globals

from DeepSpeech import create_inference_graph
from util import letters
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from util.lcs import longest_common_subsequence_general
from util.logging import log_error
from util.text import levenshtein

MAX_DIM = 10


def create_embedding_graph(input_vec_size, output_count, output_size, embedding_dim=300):
    inputs = tf.keras.Input(shape=(input_vec_size,))
    embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')(inputs)
    outputs = []
    for i in range(0, output_count):
        outputs.append(tf.layers.Dense(output_size, activation='softmax')(embedding))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='mse',
                  metrics=['mse'])

    return model


def extract_embedding_part (orig_graph,input_vec_size , embedding_dim = 300):
    embedding_input = tf.keras.Input(shape=(input_vec_size,))
    embedding_layer = tf.keras.layers.Dense(embedding_dim, activation='relu' , weights=orig_graph.layers[1].get_weights())(embedding_input)
    model = tf.keras.Model(inputs=embedding_input,  outputs=embedding_layer)
    return model


def reshape_output(row, output_chunk_number):
    # based on alphabet size 28 + 1 (no activity)
    chunk_size = Config.alphabet.size() + 1
    # Padding default
    padding = [0] * chunk_size
    padding[chunk_size - 1] = 1

    word = row[0]
    num_chunks = int(row[1])
    det_word = row[2]
    data = []
    for i in range(3, len(row)):
        data.append(float(row[i]))
    orig_chunck_count = int(len(data) / chunk_size)
    output = []

    # first step skipp the reapeated  charachters
    skipped_chunks = 0
    for i in range(0, orig_chunck_count):
        if len(output) >= output_chunk_number:
            break
        chunk = data[(i * chunk_size):((i + 1) * chunk_size )]

        if(np.argmax(chunk) == 0):
            skipped_chunks = skipped_chunks + 1
            continue

        if i == 0 or (i < len(det_word) and det_word[i] != det_word[i - 1]) or (
                orig_chunck_count - skipped_chunks) <= output_chunk_number:
            output.append(chunk)
        else:
            skipped_chunks = skipped_chunks + 1

    while len(output) < output_chunk_number:
        output.append(padding)
    return output


def load_inputs(input_file, index_file, output_chunk_number):
    print('Loading input data')
    outputs = [[] for i in range(output_chunk_number)]
    inputs = []
    words = []
    net_words = []
    letter_index = letters.load_index(index_file)

    with open(input_file, 'r') as input:
        input_csv = csv.reader(input)
        for row in input_csv:
            data = letters.word_to_feature(row[0], letter_index)
            output = reshape_output(row, output_chunk_number)
            inputs.append(data)
            words.append(row[0])
            net_words.append(row[2])
            for i in range(0, output_chunk_number):
                outputs[i].append(output[i])
    return inputs, outputs, words, net_words


def train_and_test(use_old_model = True):

    inputs, outputs, words, net_words = load_inputs('./data/logit.csv', './data/lngrams.txt', MAX_DIM)
    weights_file = './weights.dat'
    np_inputs = np.array(inputs)
    np_outputs = [np.array(o) for  o in outputs]
    graph = create_embedding_graph(len(inputs[0]), MAX_DIM, Config.alphabet.size() + 1)

    if (os.path.isfile(weights_file + '.index') and use_old_model):
        graph.load_weights(weights_file)
    else:
        graph.fit(np_inputs, np_outputs, epochs=20, steps_per_epoch=20)
        graph.save_weights(weights_file)

    embedd_graph = extract_embedding_part(graph, len(inputs[0]))

    result = graph.predict(np_inputs)
    embedd_result = embedd_graph.predict(np_inputs)

    word_alpha_leven = 0
    alpha_net_leven = 0
    word_net_leven = 0
    for i in range(len(inputs)):
        tmp = [result[j][i] for j in range(MAX_DIM)]
        alpha, logits = get_alphabet_from_logits(tmp, Config.alphabet)
        print( words[i]+' & '+ alpha + ' & ' + net_words[i] + ' \\\\ ')
        word_alpha_leven = word_alpha_leven + levenshtein(words[i], alpha)
        word_net_leven = word_net_leven + levenshtein(words[i], net_words[i])
        alpha_net_leven = alpha_net_leven + levenshtein(alpha , net_words[i])
    print('Word-Alpha Sum:' + str(word_alpha_leven))
    print('Word-Net Sum:' + str(word_net_leven))
    print('Net-Alpha Sum:' + str(alpha_net_leven))
    return embedd_graph, embedd_result , words


def find_closest(word, lgram_model, embedd_graph, embedd_corpus ):
    features = np.array([letters.word_to_feature(word, lgram_model)])
    out = embedd_graph.predict(features)[0]
    dists = []
    for  a in embedd_corpus:
        dists.append(scipy.spatial.distance.cosine(out , a))
    return np.argsort(dists)





def do_single_file_inference(input_file_path):
    with tf.Session(config=Config.session_config) as session:
        inputs, outputs = load_model(session)
        run_model(input_file_path, inputs, outputs, session)


def do_batch_file_inference(input_csv, output_file, last_processed_file=None):
    continue_flag = 0
    file_flag = 'w'
    if (last_processed_file != None):
        continue_flag = 1
        file_flag = 'w+'
    with tf.Session(config=Config.session_config) as session:
        inputs, outputs = load_model(session)
        file = pandas.read_csv(input_csv, encoding='utf-8', na_filter=False)
        with open(output_file, file_flag, newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for index, row in file.iterrows():
                if continue_flag:
                    if last_processed_file == row[0]:
                        continue_flag = 0
                        print('Continue from file :', last_processed_file)
                    else:
                        print('Skipping file :', row[0])
                    continue
                print('Processing file ', row[0])
                logits = run_model(row[0], inputs, outputs, session)
                alpha, sel_logits = get_alphabet_from_logits(logits, Config.alphabet)
                match_tuples = match_transcripts(row[2], alpha)
                for word, seg_det, start, end in match_tuples:
                    r = []
                    r.append(word)
                    r.append(end - start)
                    r.append(seg_det)
                    for x in range(start, end):
                        r.extend(sel_logits[x])
                    writer.writerow(r)


def load_model(session):
    inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)
    # Create a saver using variables from the above newly created graph
    mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
    saver = tf.train.Saver(mapping)
    # Restore variables from training checkpoint
    # TODO: This restores the most recent checkpoint, but if we use validation to counteract
    #       over-fitting, we may want to restore an earlier checkpoint.
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if not checkpoint:
        log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
        exit(1)
    checkpoint_path = checkpoint.model_checkpoint_path
    saver.restore(session, checkpoint_path)
    session.run(outputs['initialize_state'])
    return inputs, outputs


def run_model(input_file_path, inputs, outputs, session):
    features = audiofile_to_input_vector(input_file_path, Config.n_input, Config.n_context)
    num_strides = len(features) - (Config.n_context * 2)
    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2 * Config.n_context + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, Config.n_input),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)
    logits = session.run(outputs['outputs'], feed_dict={
        inputs['input']: [features],
        inputs['input_lengths']: [num_strides],
    })
    logits = np.squeeze(logits)

    return logits


def get_alphabet_from_logits(logits, alphabet):
    alpha = []
    sel_logits = []
    cnt = 0
    for ind in np.argmax(logits, 1):
        if (ind < alphabet.size()):
            alpha.append(alphabet.string_from_label(ind))
            sel_logits.append(logits[cnt])
        cnt = cnt + 1
    return ''.join(alpha), sel_logits


def match_transcripts(original_trans, detected_trans):
    # space added to include the last segment as well
    o_t = original_trans + ' '
    d_t = detected_trans + ' '
    match = longest_common_subsequence_general(o_t, d_t)
    prev = (0, 0)
    ret = []
    for x, y in match:
        if o_t[x] == ' ':
            seg_orig = o_t[prev[0]:x].strip()
            seg_det = d_t[prev[1]:y].strip()
            if len(seg_orig) == 0 or len(seg_det) == 0:
                continue
            ret.append((seg_orig, seg_det, prev[1], y))
            print('matched :', seg_orig, '<-->', seg_det)
            prev = (x + 1, y + 1)
    # returns tuples containing  the original transcript match and the detected locations
    return ret


def prepare_data():
    global FLAGS
    create_flags()
    initialize_globals()
    FLAGS.checkpoint_dir = './data/checkpoint'
    # do_single_file_inference('./data/audio/2830-3980-0043.wav')
    do_batch_file_inference('../LibriSpeech/dev-clean.csv', 'logit.csv')


def train_model():
    global FLAGS
    create_flags()
    initialize_globals()
    # prepare_data()
    embedd_graph, embedd_result, words = train_and_test()
    test_words = ['wheel' , 'buy', 'heart', 'please' , 'plug' , 'chareety']
    index_model = letters.load_index('./data/lngrams.txt')

    for tw in test_words:
        neigh = find_closest(tw, index_model , embedd_graph, embedd_result)
        #print (tw)
        close_words = set()
        index = 0
        str = tw
        while len(close_words) < 5:
            if (words[neigh[index]] in close_words):
                index = index + 1
            else:
                close_words.add(words[neigh[index]])
                #print('\t' + words[neigh[index]])
                str = str + ' & ' + words[neigh[index]]
        print (str + '\\\\')
        print('\\hline')

if __name__ == '__main__':
    train_model()
    # prepare_data()
