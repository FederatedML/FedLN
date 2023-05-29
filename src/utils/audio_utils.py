import os
import re
import tensorflow as tf
import numpy as np
import glob

def pad(waveform, sequence_length=16000):
	padding = tf.maximum(sequence_length - tf.shape(waveform)[0], 0)
	left_pad = padding // 2
	right_pad = padding - left_pad
	return tf.pad(waveform, paddings=[[left_pad, right_pad]])

def extract_window(waveform, seg_length=15690):
	waveform = pad(waveform)
	return tf.image.random_crop(waveform, [seg_length])

def extract_spectrogram(waveform, sample_rate=16000, frame_length=400, frame_step=160,  fft_length=1024, n_mels=64, fmin=60.0, fmax=7800.0):
	# A x-point STFT with frames of x ms and x% overlap.
	stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length) 
	spectrograms = tf.abs(stfts)
	# Warp the linear scale spectrograms into the mel-scale.
	num_spectrogram_bins = stfts.shape[-1]
	lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
	mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
	mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
	# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
	mel_spectrograms = tf.clip_by_value(mel_spectrograms, clip_value_min=1e-5, clip_value_max=1e8)
	log_mel_spectrograms = tf.math.log(mel_spectrograms)
	return log_mel_spectrograms[Ellipsis, tf.newaxis]

def train_prep_spcm(waveform, label, cast_to_float=True, n_mels=64):
	waveform = tf.cast(waveform, tf.float32)
	if cast_to_float: waveform/=float(tf.int16.max)
	waveform = pad(waveform)
	waveform = tf.math.l2_normalize(waveform, epsilon=1e-9)
	waveform = extract_window(waveform)
	lms = extract_spectrogram(waveform, n_mels=n_mels)
	label = tf.one_hot(label, 12)
	return lms, label

def valid_prep_spcm(waveform, label, cast_to_float=True, n_mels=64):
	waveform = tf.cast(waveform, tf.float32)
	if cast_to_float: waveform/=float(tf.int16.max)
	waveform = pad(waveform)
	waveform = tf.math.l2_normalize(waveform, axis=-1, epsilon=1e-9)
	lms = extract_spectrogram(waveform, n_mels=n_mels)
	label = tf.one_hot(label, 12)
	return lms, label

def read_tf_record(serialized_example):
	feature_description ={'audio': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64),
		'shape': tf.io.FixedLenFeature([], tf.int64), 'file_id': tf.io.FixedLenFeature([], tf.int64),
		'speaker_id': tf.io.FixedLenFeature([], tf.string),}
	example = tf.io.parse_single_example(serialized_example, feature_description)
	audio = tf.reshape(tf.io.parse_tensor(example['audio'], out_type=tf.int64), shape=[example['shape']])
	label = example['label']
	file_id = example['file_id']
	return audio, label, file_id

def remap(audio, label, file_id, lookup_labels=None, lookup_weights=None):
	if (lookup_labels is not None) and (lookup_weights is not None):
		return audio, lookup_labels.lookup(file_id), lookup_weights.lookup(file_id)
	elif (lookup_labels is not None) and (lookup_weights is None):
		return audio, lookup_labels.lookup(file_id)
	else:
		return audio, label

def load_lookup_table(data, dtype='labels'):
	if dtype == 'labels':
		return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(data[0], tf.argmax(data[1],axis=-1), key_dtype=tf.int64, value_dtype=tf.int64), default_value=-1)
	elif dtype == 'weights':
		return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(data[0], data[1], key_dtype=tf.int64, value_dtype=tf.float32), default_value=0.0)
	else:
		return None

