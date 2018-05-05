import numpy

def prune_swda_corpus_data(talks):
    print('Pruning SwDA Corpus data...\n')
    inappropriate_words = set()
    h_1 = {}
    h_2 = {}
    h_3 = {}
    h_4 = {}

    pruned_talks = []
    num_words = 0
    num_utterances = 0

    num_correct_words = 0
    num_correct_utterances = 0
    num_correct_talks = 0

    num_corrected_words = 0
    num_corrected_utterances = 0
    num_corrected_talks = 0

    num_words_failed_to_correct = 0
    num_completely_corrected_utterances = 0
    num_completely_corrected_talks = 0

    num_words_with_problematic_characters = 0
    num_utterances_with_problematic_characters = 0
    num_talks_with_problematic_characters = 0

    aints_counted = 0

    correct_sentence_endings = True
    correct_personal_pronouns_with_s = True
    correct_shortened_will_are_have = True
    correct_shortened_not = True

    for t in talks:
        num_utterances += len(t)
        pruned_content = []
        pruned_tags = []
        talk_problematic = False
        talk_corrected = False
        talk_partially_corrected = False
        talk_content = t[0]
        talk_tags = t[1]
        for j in range(len(talk_content)):
            utterance, tag = talk_content[j], talk_tags[j]

            num_words += len(utterance)
            pruned_utterance = []
            utterance_problematic = False
            utterance_corrected = False
            for w in utterance:
                word_problematic = False
                word_corrected = False
                lower_w = w.lower()
                if len(lower_w) > 0:
                    if correct_sentence_endings:
                        if (lower_w[-1] == '.') or (lower_w[-1] == ',')  or\
                           (lower_w[-1] == '?') or (lower_w[-1] == '!'):
                            word_problematic = True
                            if lower_w in h_1:
                                h_1[lower_w] += 1
                            else:
                                h_1[lower_w] = 1
                            lower_w = lower_w[:-1]
                            if lower_w.isalpha():
                                word_corrected = True
                                pruned_utterance.append(lower_w)
                            else:
                                inappropriate_words.add(lower_w)
                    if correct_personal_pronouns_with_s:
                        if lower_w == "he's" or lower_w == "she's" or lower_w == "it's":
                            word_problematic = True
                            word_corrected = True
                            if lower_w in h_4:
                                h_4[lower_w] += 1
                            else:
                                h_4[lower_w] = 1
                            pruned_utterance.append(lower_w[:-2])
                            pruned_utterance.append('is')
                    if len(lower_w) > 2:
                        if correct_shortened_will_are_have:
                            last_three = lower_w[-3:]
                            if last_three == '\'re' or last_three == '\'ll' or last_three == '\'ve':
                                word_problematic = True
                                if lower_w in h_2:
                                    h_2[lower_w] += 1
                                else:
                                    h_2[lower_w] = 1
                                second_from_last_char = lower_w[-2]
                                lower_w = lower_w[:-3]
                                if lower_w.isalpha():
                                    word_corrected = True
                                    pruned_utterance.append(lower_w)
                                    if second_from_last_char is 'l':
                                        pruned_utterance.append('will')
                                    elif second_from_last_char is 'r':
                                        pruned_utterance.append('are')
                                    else:
                                        pruned_utterance.append('have')
                                else:
                                    num_words_failed_to_correct += 1
                                    inappropriate_words.add(lower_w)
                        if correct_shortened_not:
                            if last_three == 'n\'t':
                                word_problematic = True
                                if lower_w in h_3:
                                    h_3[lower_w] += 1
                                else:
                                    h_3[lower_w] = 1
                                lower_w = lower_w[:-3]
                                if lower_w == "ai":
                                    aints_counted += 1
                                    inappropriate_words.add(lower_w)
                                else:
                                    word_corrected = True
                                    pruned_utterance.append(lower_w)
                                    pruned_utterance.append('not')
                    if not word_corrected:
                        if lower_w.isalpha():
                            pruned_utterance.append(lower_w)
                        else:
                            word_problematic = True
                            inappropriate_words.add(lower_w)
                if word_problematic:
                    utterance_problematic = utterance_problematic or word_problematic
                    num_words_with_problematic_characters += 1
                    if word_corrected:
                        utterance_corrected = utterance_corrected or word_corrected
                        num_corrected_words += 1
                    else:
                        num_words_failed_to_correct += 1
                else:
                    num_correct_words += 1

            if utterance_problematic:
                talk_problematic = True
                num_utterances_with_problematic_characters += 1
                if utterance_corrected:
                    talk_corrected = True
                    num_corrected_utterances += 1
                    if len(pruned_utterance) == len(utterance):
                        num_completely_corrected_utterances += 1
                    else:
                        talk_partially_corrected = True
            else:
                num_correct_utterances += 1

            pruned_content.append( pruned_utterance )
            pruned_tags.append( tag )
                
        if talk_problematic:
            num_talks_with_problematic_characters += 1
            if talk_corrected:
                num_corrected_talks += 1
                if not talk_partially_corrected:
                    num_completely_corrected_talks += 1
        else:
            num_correct_talks += 1
        pruned_talks.append( (pruned_content, pruned_tags) )

    print(str(len(talks)) + " talks with " + str(num_utterances) + " utterances were analyzed. A total of " + str(num_words) + " words were checked.")
    print(str(num_correct_talks) + " talks were found to be correct.")
    print("However, there were problematic characters in " + str(num_talks_with_problematic_characters) + " talks.")
    print(str(num_corrected_talks) + " of the problematic talks were attempted to be corrected.")
    print(str(num_completely_corrected_talks) + " of them were corrected completely.")

    print(str(num_correct_utterances) + " utterances were found to be correct.")
    print("However, there were problematic characters in " + str(num_utterances_with_problematic_characters) + " utterances.")
    print(str(num_corrected_utterances) + " of the problematic utterances were attempted to be corrected.")
    print(str(num_completely_corrected_utterances) + " of them were corrected completely.")

    print(str(num_correct_words) + " words were found to be correct.")
    print("However, there were problematic characters in " + str(num_words_with_problematic_characters) + " words.")
    print(str(num_corrected_words) + " of the problematic words were attempted to be corrected.")
    print(str(num_words_failed_to_correct) + " of them could not be corrected.")

    print("Also, " + str(aints_counted) + " \"ain't\"s were encountered.\n")

    print('Pruned SwDA Corpus data.')
    return pruned_talks

def vectorize_talks(talks, word_vec_dict, num_word_dimensions):
    print('Vectorizing SwDA Corpus data...')
    vectorized_talks = []
    could_not_found = []
    set_could_not_found = set()
    for t in talks:
        vectorized_content = []
        processed_tags = []
        talk_content, talk_tags = t[0], t[1]
        for j in range(len(talk_content)):
            utterance, tag = talk_content[j], talk_tags[j]
            vectorized_utterance = []
            for w in utterance:
                lower_w = w.lower()
                if lower_w in word_vec_dict:
                    vectorized_utterance.append(word_vec_dict[lower_w])
                else:
                    could_not_found.append(lower_w)
                    set_could_not_found.add(lower_w)

            vectorized_content.append(vectorized_utterance)
            processed_tags.append(tag)
        vectorized_talks.append( [vectorized_content, processed_tags] )
    print('Vectorized SwDA Corpus data.')
    return vectorized_talks

def find_longest_conversation_length(talks):
    max_conversation_length = 0
    for talk in talks:
        if max_conversation_length < len(talk[0]):
            max_conversation_length = len(talk[0])

    print('Found max_conversation_length:' + str(max_conversation_length))
    return max_conversation_length

def find_max_utterance_length(talks):
    max_utterance_length = 0
    for talk in talks:
        for utterance in talk[0]:
            if max_utterance_length < len(utterance):
                max_utterance_length = len(utterance)

    print('Found max_utterance_length:' + str(max_utterance_length))
    return max_utterance_length

def arrange_word_to_vec_dict(talks, word_vec_dict, num_word_dimensions):
    seen_words = set()
    # From a set of seen words
    for talk in talks:
        for utterance in talk[0]:
            for word in utterance:
                lowercase_word = word.lower()
                seen_words.add(lowercase_word)

    # Remove words that are not in the dataset
    to_be_deleted = []
    for word in word_vec_dict.keys():
        if word not in seen_words:
            to_be_deleted.append(word)

    for word in to_be_deleted:
        del word_vec_dict[word]

    # Add random vectors for words that are not seen in the embedding
    for word in seen_words:
        if word not in word_vec_dict:
            word_vec_dict[word] = numpy.zeros(num_word_dimensions)

def form_word_to_index_dict_from_dataset(word_vec_dict):
    word_to_index = {}
    next_index_to_assign = 1
    for key in word_vec_dict.keys():
        word_to_index[key] = next_index_to_assign
        next_index_to_assign += 1
    return word_to_index

def prepare_data(dataset_loading_function, dataset_file_path,
                 embedding_loading_function, embedding_file_path):
    # Read dataset
    read_talks, talk_names, tag_indices, tag_occurances = dataset_loading_function(dataset_file_path)
    num_tags = len(tag_indices.keys())

    #Prune word data
    talks = prune_swda_corpus_data(read_talks)
    read_talks.clear()

    # Read vector representation of words
    num_words, num_word_dimensions, word_vec_dict = embedding_loading_function(embedding_file_path)

    # Transform words in dataset to vectors
    vectorized_talks = vectorize_talks(talks, word_vec_dict, num_word_dimensions)
    talks.clear()
    word_vec_dict.clear()

    max_utterance_len = find_max_utterance_length(vectorized_talks)
    max_conversation_len = find_longest_conversation_length(vectorized_talks)

    return (vectorized_talks, talk_names, tag_indices), (max_conversation_len, max_utterance_len,
                                                         num_word_dimensions, num_tags)

