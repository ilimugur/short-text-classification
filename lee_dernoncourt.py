import copy
import random
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from train_set_preferences import swda_train_set_idx, swda_valid_set_idx, swda_test_set_idx
from train_set_preferences import mrda_train_set_idx, mrda_valid_set_idx, mrda_test_set_idx
from helpers import find_max_utterance_length, find_longest_conversation_length
from helpers import form_datasets, find_unique_words_in_dataset
from helpers import vectorize_talks, form_word_vec_dict
from translate import read_translated_swda_corpus_data

from fastText_multilingual.fasttext import FastVector

def lee_dernoncourt_batch_generator(dataset_x, dataset_y, timesteps, num_word_dimensions, num_tags):
    # Create empty arrays to contain batch of features and labels#
    index_list = [x for x in range(len(dataset_x))]
    index_list.append(index_list[0])
    random.shuffle(index_list)

    k = -1
    while True:
        k = (k + 1) % len(index_list)
        index = index_list[k]
        num_utterances = len(dataset_x[index])
        utterances = dataset_x[index]

        batch_features = numpy.zeros((num_utterances, timesteps, num_word_dimensions))
        batch_labels = to_categorical(dataset_y[index], num_tags)

        for i in range(num_utterances):
            utterance = copy.deepcopy(utterances[i])
            num_to_append = max(0, timesteps - len(utterance))
            if num_to_append > 0:
                appendage = [numpy.zeros(num_word_dimensions)] * num_to_append
                utterance += appendage

            batch_features[i] = utterance
            del utterance

        yield batch_features, batch_labels


def prepare_lee_dernoncourt_model(max_conversation_len, timesteps, num_word_dimensions, num_tags,
                                  loss_function, optimizer):
    #Hyperparameters
    n = 100

    model = Sequential()
    model.add(LSTM(n, return_sequences = True, input_shape = (timesteps, num_word_dimensions)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_tags, activation='tanh'))
    model.add(Dense(num_tags, activation='softmax'))
    model.compile(loss = loss_function,
                  optimizer = optimizer,
                  metrics=['accuracy'])
    return model

def train_lee_dernoncourt(model, training, validation, num_epochs_to_train,
                          timesteps, num_word_dimensions, num_tags):
    early_stop = EarlyStopping(monitor='val_loss', patience = 10)

    num_training_steps = len(training[0])
    num_validation_steps = len(validation[0])
    model.fit_generator(lee_dernoncourt_batch_generator(training[0], training[1],
                                                        timesteps, num_word_dimensions, num_tags),
                        steps_per_epoch = num_training_steps,
                        epochs = num_epochs_to_train,
                        validation_data = lee_dernoncourt_batch_generator(validation[0],
                                                                          validation[1],
                                                                          timesteps,
                                                                          num_word_dimensions,
                                                                          num_tags),
                        validation_steps = num_validation_steps,
                        callbacks = [early_stop])
    return model

def evaluate_lee_dernoncourt(model, testing, timesteps, num_word_dimensions, num_tags):
    num_testing_steps = len(testing[0])
    score = model.evaluate_generator(lee_dernoncourt_batch_generator(testing[0], testing[1],
                                                                     timesteps, num_word_dimensions,
                                                                     num_tags),
                                     steps = num_testing_steps)
    return score[1]

def lee_dernoncourt(dataset, dataset_loading_function, dataset_file_path,
                    embedding_loading_function, 
                    source_lang, source_lang_embedding_file, source_lang_transformation_file,
                    target_lang, target_lang_embedding_file, target_lang_transformation_file,
                    translation_set_file,
                    src_word_set,
                    translated_pairs_file,
                    translated_word_dict,
                    translation_complete,
                    target_test_data_path,
                    num_epochs_to_train, loss_function, optimizer,
                    shuffle_words, load_from_model_file, previous_training_epochs,
                    save_to_model_file):
    monolingual = target_lang is None

    # Read dataset
    talks_read, talk_names, tag_indices, tag_occurances = dataset_loading_function(dataset_file_path)
    if dataset == 'MRDA':
        uninterpretable_label_index = tag_indices['z']
        train_set_idx, valid_set_idx, test_set_idx = mrda_train_set_idx, mrda_valid_set_idx,\
                                                     mrda_test_set_idx
    elif dataset == 'SwDA':
        uninterpretable_label_index = tag_indices['%']
        train_set_idx, valid_set_idx, test_set_idx = swda_train_set_idx, swda_valid_set_idx,\
                                                     swda_test_set_idx
    else:
        print("Dataset unknown!")
        exit(0)

    num_tags = len(tag_indices.keys())

    if not monolingual:
        talks_read, talk_names = read_translated_swda_corpus_data(dataset, talks_read, talk_names,
                                                                  target_test_data_path, target_lang)

    #Prune word data
#    talks_read_initial = talks_read
#    talks_read = prune_swda_corpus_data(talks_read_initial)
#    talks_read_initial.clear()
    for k, c in enumerate(talks_read):
        for u in c[0]:
            for i, word in enumerate(u):
                u[i] = word.rstrip(',').rstrip('.').rstrip('?').rstrip('!')

    if src_word_set is None:
        src_word_set = find_unique_words_in_dataset(talks_read, talk_names, test_set_idx,
                                                    monolingual, translation_set_file)

    if not monolingual:
        target_word_set = find_unique_words_in_dataset(talks_read, talk_names, test_set_idx,
                                                       monolingual, include_idx_set_members = True)
    else:
        target_word_set = None

    word_vec_dict = form_word_vec_dict(dataset, talks_read, talk_names, monolingual,
                                       src_word_set, target_word_set,
                                       translated_word_dict, translated_pairs_file,
                                       source_lang_embedding_file, target_lang_embedding_file,
                                       source_lang_transformation_file,
                                       target_lang_transformation_file,
                                       translation_complete)

    for word, vector in word_vec_dict.items():
        num_word_dimensions = len(vector)
        break

    # Transform words in dataset to vectors
    vectorized_talks = vectorize_talks(talks_read, word_vec_dict, num_word_dimensions)
    talks_read.clear()
    word_vec_dict.clear()

    timesteps = find_max_utterance_length(vectorized_talks)
    max_conversation_len = find_longest_conversation_length(vectorized_talks)

    training, validation, testing = form_datasets(vectorized_talks, talk_names,
                                                  test_set_idx, valid_set_idx,
                                                  train_set_idx)
    vectorized_talks.clear()
    talk_names.clear()

    if shuffle_words:
        for talk in training[0]:
            for utterance in talk:
                random.shuffle(utterance)

    if load_from_model_file:
        model = load_model(load_from_model_file)
    else:
        model = prepare_lee_dernoncourt_model(max_conversation_len, timesteps, num_word_dimensions,
                                              num_tags, loss_function, optimizer)

    if num_epochs_to_train > 0:
        train_lee_dernoncourt(model, training, validation, num_epochs_to_train,
                              timesteps, num_word_dimensions, num_tags)
        if save_to_model_file:
            model.save(save_to_model_file)

    print("EVALUATING...")
    score = evaluate_lee_dernoncourt(model, testing, timesteps, num_word_dimensions, num_tags)
    print("Accuracy: " + str(score * 100) + "%")

    return model


