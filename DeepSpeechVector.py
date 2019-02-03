#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
import os
import sys

import pandas


import evaluate
import numpy as np
import tensorflow as tf

from six.moves import range
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import log_error
from util.preprocess import preprocess
from util.lcs import longest_common_subsequence_general

from DeepSpeech import create_inference_graph

# Logging
# =======







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



if __name__ == '__main__':
    create_flags()
    initialize_globals()
    FLAGS.checkpoint_dir = './data/checkpoint'
    # do_single_file_inference('./data/audio/2830-3980-0043.wav')
    do_batch_file_inference('../LibriSpeech/dev-clean.csv', 'logit.csv' )
