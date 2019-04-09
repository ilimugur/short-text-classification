import numpy

def read_word2vec(file_name):
    print('Reading word2vec data...')
    max_w = 50
    with open(file_name, 'rb') as f:
        # Read num words
        # Read num dimensions of each word
        num_words = int.from_bytes(f.read(4)) # alternatively: struct.unpack('i', fin.read(4))
        num_dims = int.from_bytes(f.read(4))
        word_2_vec = {}
        for b in range(0, num_words):
            a = 0
            word = ''
            while True:
                new_char = f.read(1)
                if new_char  == '' or new_char == ' ':
                    break
                if new_char != '\n':
                    word.append(new_char)
                    if a < max_w:
                        a += 1
            word_vec = []
            for i in range(num_dims):
                word_vec.append(float.from_bytes(f.read(4)))
            word_2_vec[word] = word_vec
    print('Read word2vec data.')
    return (num_words, num_dimensions, word_2_vec)

def read_fasttext_embedding(embedding_file):
    print('Reading FastText embedding...')
    word_2_vec = {}

    with open(embedding_file, 'r') as f:
        (num_words, num_dimensions) = \
            (int(x) for x in f.readline().rstrip('\n').split(' '))
        for line in f:
            elems = line.rstrip('\n').split(' ')
            try:
                tmp_list = [float(dim_val) for dim_val in elems[1: num_dimensions + 1]]
            except:
                print(elems)
                raise
            word_2_vec[ elems[0] ] = numpy.array(tmp_list)

    print('Read FastText embedding.')
    return (num_words, num_dimensions, word_2_vec)

def read_glove_twitter(file_name):
    print('Reading GloVe Twitter data...')
    f = open(file_name, 'r')
    num_words = 0
    word_2_vec = {}
    word_vec = []
    while True:
        line = f.readline()
        if line == '':
            break
        read_values = line.split(' ')
        del line
        word = read_values[0]
        tmp_list = [float(dim_val) for dim_val in read_values[1:]]
        word_vec = numpy.array(tmp_list)
        tmp_list.clear()
        read_values.clear()
        word_2_vec[word] = word_vec
        num_words += 1
    num_dimensions = len(word_vec)
    f.close()

    print('Read GloVe Twitter data.')
    return (num_words, num_dimensions, word_2_vec)

