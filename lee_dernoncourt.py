import copy
import random
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.utils import to_categorical
from train_set_preferences import valid_set_idx, test_set_idx
from helpers import prepare_data

def form_datasets(vectorized_talks, talk_names, max_sentence_length, word_dimensions):
    print('Forming dataset appropriately...')
    
    x_train_list = []
    y_train_list = []
    x_valid_list = []
    y_valid_list = []
    x_test_list = []
    y_test_list = []
    t_i = 0
    for i in range(len(vectorized_talks)):
        vt = vectorized_talks[i]
        if talk_names[i] in test_set_idx:
            x_test_list.append( vt[0] )
            y_test_list.append( vt[1] )
        if talk_names[i] in valid_set_idx:
            x_valid_list.append( vt[0] )
            y_valid_list.append( vt[1] )
        else:
            x_train_list.append( vt[0] )
            y_train_list.append( vt[1] )
        t_i += 1

    print('Formed dataset appropriately.')
    return ((x_train_list, y_train_list), (x_valid_list, y_valid_list), (x_test_list, y_test_list))


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


def prepare_lee_dernoncourt_model(timesteps, num_word_dimensions, num_tags,
                                  loss_function, optimizer):
    #Hyperparameters
    n = 100

    model = Sequential()
    model.add(LSTM(n, return_sequences = True, input_shape = (timesteps, num_word_dimensions)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_tags, input_shape=(n, ), activation='tanh'))
    model.add(Dense(num_tags, input_shape=(num_tags, ), activation='softmax'))
    model.compile(loss = loss_function,
                  optimizer = optimizer,
                  metrics=['accuracy'])
    return model

def train_lee_dernoncourt(model, training, validation, timesteps, num_word_dimensions, num_tags):
    num_training_steps = len(training[0])
    num_validation_steps = len(validation[0])
    model.fit_generator(lee_dernoncourt_batch_generator(training[0], training[1],
                                                        timesteps, num_word_dimensions, num_tags),
                        steps_per_epoch = num_training_steps,
                        epochs = 10,
                        validation_data = lee_dernoncourt_batch_generator(validation[0],
                                                                          validation[1],
                                                                          timesteps,
                                                                          num_word_dimensions,
                                                                          num_tags),
                        validation_steps = num_validation_steps)
    return model

def evaluate_lee_dernoncourt(model, testing, timesteps, num_word_dimensions, num_tags):
    num_testing_steps = len(testing[0])
    score = model.evaluate_generator(lee_dernoncourt_batch_generator(testing[0], testing[1],
                                                                     timesteps, num_word_dimensions,
                                                                     num_tags),
                                     steps = num_testing_steps)
    return score

def lee_dernoncourt(dataset_loading_function, dataset_file_path,
                  embedding_loading_function, embedding_file_path,
                  loss_function, optimizer):
    data, dimensions = prepare_data(dataset_loading_function, dataset_file_path,
                                    embedding_loading_function, embedding_file_path)
    (vectorized_talks, talk_names), (timesteps, num_word_dimensions, num_tags) = (data, dimensions)
    training, validation, testing = form_datasets(vectorized_talks, talk_names,
                                                      timesteps, num_word_dimensions)
    vectorized_talks.clear()
    talk_names.clear()

    model = prepare_lee_dernoncourt_model(timesteps, num_word_dimensions, num_tags,
                                          loss_function, optimizer)
    train_lee_dernoncourt(model, training, validation, timesteps, num_word_dimensions, num_tags)
    score = evaluate_lee_dernoncourt(model, testing, timesteps, num_word_dimensions, num_tags)
    print(score)
    return model


