import copy
import random
import sys
import numpy
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.models import load_model
from keras.utils import to_categorical
from train_set_preferences import mrda_valid_set_idx, mrda_test_set_idx, mrda_train_set_idx
from train_set_preferences import swda_valid_set_idx, swda_test_set_idx, swda_train_set_idx
from translate import read_translated_swda_corpus_data
from helpers import arrange_word_to_vec_dict, form_word_to_index_dict_from_dataset
from helpers import find_max_utterance_length, find_longest_conversation_length
from helpers import write_word_translation_dict_to_file, write_word_set_to_file
from helpers import pad_dataset_to_equal_length
from helpers import form_datasets, find_unique_words_in_dataset
from helpers import form_word_vec_dict

from fastText_multilingual.fasttext import FastVector

def form_mini_batches(dataset_x, max_mini_batch_size):
    num_conversations = len(dataset_x)

    # Form mini batches of equal-length conversations
    mini_batches = {}
    for i in range(num_conversations):
        num_utterances = len(dataset_x[i])
        if num_utterances in mini_batches:
            mini_batches[num_utterances].append( i )
        else:
            mini_batches[num_utterances] = [ i ]

    # Enforce max_batch_size on previously formed mini batches
    mini_batch_list = []
    for conversations in mini_batches.values():
        mini_batch_list += [conversations[x: x + max_mini_batch_size] for x in range(0, len(conversations), max_mini_batch_size)]

    return mini_batch_list


def kadjk_batch_generator(dataset_x, dataset_y, tag_indices,
                          mini_batch_list, max_conversation_length,
                          timesteps, num_word_dimensions, num_tags,
                          word_index_to_append, tag_index_to_append):
    num_mini_batches = len(mini_batch_list)

    # Shuffle the order of batches
    index_list = [x for x in range(num_mini_batches)]
    random.shuffle(index_list)

    total = 0

    k = -1
    while True:
        k = (k + 1) % len(index_list)
        index = index_list[k]
        conversation_indices = mini_batch_list[index]

        num_conversations = len(conversation_indices)
        batch_features = numpy.empty(shape = (num_conversations, max_conversation_length, timesteps),
                                     dtype = int)
        label_list = []

        for i in range(num_conversations):
            utterances = dataset_x[conversation_indices[i]]
            labels = copy.deepcopy(dataset_y[conversation_indices[i]])
            num_utterances = len(utterances)
            num_labels_to_append = max(0, max_conversation_length - len(labels))
            labels += [tag_index_to_append] * num_labels_to_append
            tags = to_categorical(labels, num_tags)
            del labels

            for j in range(num_utterances):
                utterance = copy.deepcopy(utterances[j])
                num_to_append = max(0, timesteps - len(utterance))
                if num_to_append > 0:
                    appendage = [word_index_to_append] * num_to_append
                    utterance += appendage

                batch_features[i][j] = utterance
                del utterance

            remaining_space = (max_conversation_length - num_utterances, timesteps)
            batch_features[i][num_utterances:] = numpy.ones(remaining_space) * word_index_to_append
            label_list.append(tags)

        batch_labels = numpy.array(label_list)
        del label_list

        total += 1
        yield batch_features, batch_labels



def prepare_kadjk_model(max_mini_batch_size,
                        max_conversation_length, timesteps, num_word_dimensions,
                        word_to_index, word_vec_dict,
                        num_tags, loss_function, optimizer):
    #Hyperparameters
    m = timesteps
    h = timesteps

    model = Sequential()

    dictionary_size = len(word_to_index) + 1
    print('dictionary_size:' + str(dictionary_size))

    embedding_weights = numpy.zeros((dictionary_size, num_word_dimensions))
    for word, index in word_to_index.items():
        embedding_weights[index, :] = word_vec_dict[word]

    # define inputs here
    embedding_layer = Embedding(dictionary_size, num_word_dimensions,
                                weights=[embedding_weights],
                                embeddings_regularizer=regularizers.l2(0.0001))
    model.add(TimeDistributed(embedding_layer,
                              input_shape=(max_conversation_length, timesteps)))

#    model.add(TimeDistributed(Bidirectional(LSTM(m // 2, return_sequences=True,
#                                            kernel_regularizer=regularizers.l2(0.0001)))))
#    model.add(TimeDistributed(Dropout(0.2)))
#    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(TimeDistributed(Bidirectional(LSTM(m // 2,
                                            kernel_regularizer=regularizers.l2(0.0001)))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(h // 2, return_sequences = True,
                                 kernel_regularizer=regularizers.l2(0.0001)), merge_mode='concat'))
    model.add(Dropout(0.2))
    crf = CRF(num_tags, sparse_target=False, kernel_regularizer=regularizers.l2(0.0001))
    print("Before CRF: %s" % str(model.output_shape))
    model.add(crf)
    model.compile(optimizer, loss = crf_loss,
                  metrics=[crf_accuracy])
    #TODO: Can we support providing custom loss functions like Lee-Dernoncourt model?
    return model

epochs_trained_so_far = 0

def learning_rate_scheduler(epoch, lr):
    e = epoch + epochs_trained_so_far
    new_lr = 1.0 / (2.0 ** (e // 5))
    print('XXXXXXXX this epoch:%d\toverall epoch:%d\t, lr:%f' % (epoch, e, new_lr))
    return new_lr
    if e % 5 == 0:
        if e > 0:
            return lr * 0.5
        else:
            return 1.0
    return lr

def train_kadjk(model, training, validation, num_epochs_to_train, tag_indices, max_mini_batch_size,
                max_conversation_length, timesteps, num_word_dimensions, num_tags,
                end_of_line_word_index, uninterpretable_label_index):
    training_mini_batch_list = form_mini_batches(training[0], max_mini_batch_size)
    validation_mini_batch_list = form_mini_batches(validation[0], max_mini_batch_size)

    num_training_steps = len(training_mini_batch_list)
    num_validation_steps = len(validation_mini_batch_list)

    early_stop = EarlyStopping(monitor='val_loss', patience = 5)
    change_learning_rate = LearningRateScheduler(learning_rate_scheduler)

    train_generator = kadjk_batch_generator(training[0], training[1], tag_indices,
                                            training_mini_batch_list, max_conversation_length,
                                            timesteps, num_word_dimensions, num_tags,
                                            end_of_line_word_index, uninterpretable_label_index)
    validation_generator = kadjk_batch_generator(validation[0], validation[1],
                                                 tag_indices,
                                                 validation_mini_batch_list, 
                                                 max_conversation_length, timesteps,
                                                 num_word_dimensions, num_tags,
                                                 end_of_line_word_index,
                                                 uninterpretable_label_index)
    print("num_training_steps: %d " % num_training_steps)
    print("num_validation_steps: %d " % num_validation_steps)
    model.fit_generator(train_generator,
                        steps_per_epoch = num_training_steps,
                        epochs = num_epochs_to_train,
                        validation_data = validation_generator,
                        validation_steps = num_validation_steps,
                        callbacks = [early_stop, change_learning_rate])
    return model

def evaluate_kadjk(model, testing, tag_indices, max_mini_batch_size, max_conversation_length,
                   timesteps, num_word_dimensions, num_tags,
                   end_of_line_word_index, uninterpretable_label_index):
    testing_mini_batch_list = form_mini_batches(testing[0], max_mini_batch_size)
    num_testing_steps = len(testing_mini_batch_list)
    print("num_testing_steps: %d " % num_testing_steps)
    generator = kadjk_batch_generator(testing[0], testing[1],
                                      tag_indices,
                                      testing_mini_batch_list, 
                                      max_conversation_length, timesteps,
                                      num_word_dimensions, num_tags,
                                      end_of_line_word_index,
                                      uninterpretable_label_index)
    score = model.evaluate_generator(generator,
                                     steps = num_testing_steps)
    print(str(model.metrics_names))
    print(str(score))

    return score[1]

def kadjk(dataset, dataset_loading_function, dataset_file_path,
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
    global epochs_trained_so_far
    monolingual = target_lang is None

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

    if not monolingual:
        read_translated_swda_corpus_data(dataset, talks_read, talk_names, target_test_data_path, target_lang)

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

    print("Translated conversation dataset.")

    arrange_word_to_vec_dict(talks_read, talk_names, source_lang, target_lang, word_vec_dict, num_word_dimensions)
    word_to_index = form_word_to_index_dict_from_dataset(word_vec_dict)

    print("Dataset arranged.")

    end_of_line_word = '<unk>'
    end_of_line_word_index = len(word_to_index) + 1
    word_to_index[end_of_line_word] = end_of_line_word_index
    word_vec_dict[end_of_line_word] = numpy.random.random(num_word_dimensions)

    talks = [([[word_to_index[w.lower()] for w in u] for u in c[0]], c[1]) for k, c in enumerate(talks_read)]
    talks_read.clear()

    timesteps = find_max_utterance_length(talks)
    max_conversation_length = find_longest_conversation_length(talks)
    num_tags = len(tag_indices.keys())

    training, validation, testing = form_datasets(talks, talk_names,
                                                  test_set_idx, valid_set_idx, train_set_idx)
    talk_names.clear()
    talks.clear()

    print("Training, validation and tesing datasets are formed.")

    if shuffle_words:
        for talk in training[0]:
            for utterance in talk:
                random.shuffle(utterance)

    pad_dataset_to_equal_length(training, timesteps)
    pad_dataset_to_equal_length(validation, timesteps)
    pad_dataset_to_equal_length(testing, timesteps)

    print("Checking indices of word_to_index:")
    index_to_word = {val:key for key, val in word_to_index.items()}
    for i in range(0, len(word_to_index)):
        if i not in index_to_word:
            print(str(i))

    max_mini_batch_size = 64

    print("Previous training epochs:%d" % previous_training_epochs)
    if load_from_model_file is not None:
        epochs_trained_so_far = previous_training_epochs
        custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}
        model = load_model(load_from_model_file, custom_objects)
        print("Loaded the model.")
    else:
        model = prepare_kadjk_model(max_mini_batch_size, max_conversation_length,
                                    timesteps, num_word_dimensions, word_to_index,
                                    word_vec_dict, num_tags, loss_function,
                                    optimizer)
        print("Prepared the model.")
    print('word_vec_dict:' + str(len(word_vec_dict)))
    print('word_to_index:' + str(len(word_to_index)))

    word_vec_dict.clear()
    word_to_index.clear()

    if num_epochs_to_train > 0:
        print("BEGINNING THE TRAINING...")
        train_kadjk(model, training, validation, num_epochs_to_train, tag_indices,
                    max_mini_batch_size, max_conversation_length,
                    timesteps, num_word_dimensions, num_tags,
                    end_of_line_word_index, uninterpretable_label_index)
        if save_to_model_file:
            model.save(save_to_model_file)

    print("EVALUATING...")

    score = evaluate_kadjk(model, testing, tag_indices, max_mini_batch_size,
                           max_conversation_length, timesteps,
                           num_word_dimensions, num_tags,
                           end_of_line_word_index, uninterpretable_label_index)

    print("Accuracy: %s" % str(score * 100) + "%")

    return model
