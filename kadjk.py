import copy
import random
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras.utils import to_categorical
from train_set_preferences import valid_set_idx, test_set_idx
from helpers import prepare_data
from helpers import arrange_word_to_vec_dict, form_word_to_index_dict_from_dataset
from helpers import find_max_utterance_length, find_longest_conversation_length

def form_datasets(talks, talk_names, max_sentence_length, word_dimensions):
    print('Forming dataset appropriately...')
    
    x_train_list = []
    y_train_list = []
    x_valid_list = []
    y_valid_list = []
    x_test_list = []
    y_test_list = []
    t_i = 0
    for i in range(len(talks)):
        t = talks[i]
        if talk_names[i] in test_set_idx:
            x_test_list.append( t[0] )
            y_test_list.append( t[1] )
        if talk_names[i] in valid_set_idx:
            x_valid_list.append( t[0] )
            y_valid_list.append( t[1] )
        else:
            x_train_list.append( t[0] )
            y_train_list.append( t[1] )
        t_i += 1

    print('Formed dataset appropriately.')
    return ((x_train_list, y_train_list), (x_valid_list, y_valid_list), (x_test_list, y_test_list))

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

    f = open('error.log', 'w')
    for i in range(dictionary_size):
        s = ''
        for j in range(num_word_dimensions):
            s += str(embedding_weights[i][j]) + " "
        f.write(s + '\n')
    f.close()

    # define inputs here
    embedding_layer = Embedding(dictionary_size, num_word_dimensions,
                                weights=[embedding_weights], mask_zero=True)
    model.add(TimeDistributed(embedding_layer,
                              input_shape=(max_conversation_length, timesteps)))

    model.add(TimeDistributed(Bidirectional(LSTM(m // 2, return_sequences=True))))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(GlobalMaxPooling1D()))
    model.add(Bidirectional(LSTM(h // 2, return_sequences = True), merge_mode='concat'))
    model.add(Dropout(0.2))
    crf = CRF(num_tags, sparse_target=False)
    model.add(crf)
    model.compile('adam', loss = crf.loss_function,
                  metrics=[crf.accuracy])
    #TODO: Can we support providing custom loss functions and optimizers like Lee-Dernoncourt model?
    return model

def learning_rate_scheduler(epoch, lr):
    if epoch % 5 == 0:
        if epoch > 0:
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

    early_stop = EarlyStopping(patience = 5)
    change_learning_rate = LearningRateScheduler(learning_rate_scheduler)

    model.fit_generator(kadjk_batch_generator(training[0], training[1], tag_indices,
                                              training_mini_batch_list, max_conversation_length,
                                              timesteps, num_word_dimensions, num_tags,
                                              end_of_line_word_index, uninterpretable_label_index),
                        steps_per_epoch = num_training_steps,
                        epochs = num_epochs_to_train,
                        validation_data = kadjk_batch_generator(validation[0], validation[1],
                                                                tag_indices,
                                                                validation_mini_batch_list, 
                                                                max_conversation_length, timesteps,
                                                                num_word_dimensions, num_tags,
                                                                end_of_line_word_index,
                                                                uninterpretable_label_index),
                        validation_steps = num_validation_steps,
                        callbacks = [early_stop, change_learning_rate])
    return model

def evaluate_kadjk(model, testing, tag_indices, max_mini_batch_size, max_conversation_length,
                   timesteps, num_word_dimensions, num_tags,
                   end_of_line_word_index, uninterpretable_label_index):
    testing_mini_batch_list = form_mini_batches(testing[0], max_mini_batch_size)
    num_testing_steps = len(testing_mini_batch_list)
    score = model.evaluate_generator(kadjk_batch_generator(testing[0], testing[1],
                                                           tag_indices,
                                                           testing_mini_batch_list, 
                                                           max_conversation_length, timesteps,
                                                           num_word_dimensions, num_tags,
                                                           end_of_line_word_index,
                                                           uninterpretable_label_index),
                                     steps = num_testing_steps)
    return score[1]

def kadjk(dataset_loading_function, dataset_file_path,
                    embedding_loading_function, embedding_file_path,
                    num_epochs_to_train,
                    loss_function, optimizer, load_model_from_file, save_model, model_filename):
    talks_read, talk_names, tag_indices, tag_occurances = dataset_loading_function(dataset_file_path)

    num_words, num_word_dimensions, word_vec_dict = embedding_loading_function(embedding_file_path)

    timesteps = find_max_utterance_length(talks_read)
    max_conversation_length = find_longest_conversation_length(talks_read)
    num_tags = len(tag_indices.keys())


    arrange_word_to_vec_dict(talks_read, word_vec_dict, num_word_dimensions)
    word_to_index = form_word_to_index_dict_from_dataset(word_vec_dict)

    end_of_line_word = '<unk>'
    end_of_line_word_index = len(word_to_index) + 1
    word_to_index[end_of_line_word] = end_of_line_word_index
    word_vec_dict[end_of_line_word] = numpy.random.random(num_word_dimensions)

    uninterpretable_label_index = tag_indices['%']

    talks = [([[word_to_index[w.lower()] for w in u] for u in c[0]], c[1]) for c in talks_read]
    talks_read.clear()

    training, validation, testing = form_datasets(talks, talk_names, timesteps, num_word_dimensions)
    talk_names.clear()

    for i in range(len(training)):
        if len(training[0][i]) != len(training[1][i]):
            print("Found it: " + str(i))
            return

    max_mini_batch_size = 64
    model = prepare_kadjk_model(max_mini_batch_size, max_conversation_length,
                                timesteps, num_word_dimensions, word_to_index, word_vec_dict,
                                num_tags, loss_function, optimizer)
    print('word_vec_dict:' + str(len(word_vec_dict)))
    print('word_to_index:' + str(len(word_to_index)))

    print("Checking indices of word_to_index:")
    index_to_word = {val:key for key, val in word_to_index.items()}
    for i in range(0, len(word_to_index)):
        if i not in index_to_word:
            print(str(i))
    print("Checked indices of word_to_index.")

    word_vec_dict.clear()
    word_to_index.clear()

    if load_model_from_file is not None:
        save_load_utils.load_all_weights(model, model_filename)

    if num_epochs_to_train > 0:
        train_kadjk(model, training, validation, num_epochs_to_train, tag_indices,
                    max_mini_batch_size, max_conversation_length,
                    timesteps, num_word_dimensions, num_tags,
                    end_of_line_word_index, uninterpretable_label_index)

    score = evaluate_kadjk(model, testing, tag_indices, max_mini_batch_size,
                           max_conversation_length, timesteps,
                           num_word_dimensions, num_tags,
                           end_of_line_word_index, uninterpretable_label_index)

    print("Accuracy: " + str(score * 100) + "%")

    if save_model:
        save_load_utils.save_all_weights(model, model_filename)
    return model

