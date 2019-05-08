import numpy as np
from fastText_multilingual.fasttext import FastVector

# Training code for translation matrices

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def import_expert_signal(file_path):
    word_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            word_pair_list = line.split()
            if len(word_pair_list) == 2:
                word_pair_list.reverse() # exper signal word pairs are given in reverse
                word_pairs.append(tuple(word_pair_list))
    return word_pairs
            

def save_trained_matrix_to_file(matrix_path, matrix):
    with open(matrix_path, 'w') as f:
        for i in range(matrix.shape[0]):
            s = np.format_float_scientific(matrix[i][0], unique=False, precision=18)
            for j in range(1, matrix.shape[1]):
                s += ' %s' % np.format_float_scientific(matrix[i][j], unique=False, precision=18)
            f.write('%s\n' % s)

source_languages = ['de', 'es', 'tr']
datasets = [('fasttextwiki/', 'wiki.%s.vec'),
             ('conll17word2vec/', 'conll17.%s.txt'),
             ('fasttext157/', 'cc.%s.300.vec')]
dictionaries = ['expert', 'automated']

for prefix, file_format in datasets:
    monolingual_language_files_path = '../word-embeddings/%smonolingual/' % prefix
    target_dictionary = FastVector(vector_file=monolingual_language_files_path + (file_format % 'en') )
    target_words = set(target_dictionary.word2id.keys())

    for signal in dictionaries:
        training_matrices_path = ('alignment_matrices/%s' % prefix) + signal + '/%s.txt'
        dimension = None
        for language in source_languages:
            source_dictionary = FastVector(vector_file=monolingual_language_files_path +
                                                       (file_format % language))

            source_words = set(source_dictionary.word2id.keys())

            if signal is 'automated':
                # For pseudo-dictionary training
                overlap = list(target_words & source_words)
                bilingual_dictionary = [(entry, entry) for entry in overlap]
            else:
                # For expert dictionary training
                expert_dictionary_path = '../word-embeddings/%sexpert/train/dict.en.%s.txt' %\
                                         (prefix, language)
                bilingual_dictionary = import_expert_signal(expert_dictionary_path)

            source_matrix, target_matrix = make_training_matrices(
                source_dictionary, target_dictionary, bilingual_dictionary)

            del source_dictionary

            transform = learn_transformation(source_matrix, target_matrix)
            dimension = transform.shape[0]

            save_file = training_matrices_path % language
            save_trained_matrix_to_file(save_file, transform)
            print('Saved training matrix for \'%s\' to %s'% (language, save_file))
        target_transform = np.identity(dimension, dtype = float)
        save_file = training_matrices_path % 'en'
        save_trained_matrix_to_file(training_matrices_path % 'en', target_transform)
        print('Saved training matrix for \'en\' to %s' % save_file)
