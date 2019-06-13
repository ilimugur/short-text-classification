import numpy
from fastText_multilingual.fasttext import FastVector
from train_set_preferences import swda_test_set_idx, mrda_test_set_idx

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
        if max_conversation_length < len(talk[1]):
            max_conversation_length = len(talk[1])

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

def arrange_word_to_vec_dict(talks, talk_names, source_lang, target_lang,
                             word_vec_dict, num_word_dimensions):
    seen_words = set()
    # Form a set of seen words
    for k, talk in enumerate(talks):
        for utterance in talk[0]:
            for word in utterance:
                seen_words.add(word.lower())

    # Remove words that are not in the dataset
    to_be_deleted = []
    for word in word_vec_dict:
        if word not in seen_words:
            to_be_deleted.append(word)

    for word in to_be_deleted:
        del word_vec_dict[word]

    print("found %d unnecessary words in word_vec_dict." % len(to_be_deleted))

    # Add random vectors for words that are not seen in the embedding
    for word in seen_words:
        if word not in word_vec_dict:
            word_vec_dict[word] = numpy.zeros(num_word_dimensions)

def form_word_to_index_dict_from_dataset(word_vec_dict):
    word_to_index = {}
    next_index_to_assign = 1
    for key in sorted(word_vec_dict.keys()):
        word_to_index[key] = next_index_to_assign
        next_index_to_assign += 1
    return word_to_index

# Does this support reading/writing unicode characters?
def read_word_set_from_file(file_path):
    word_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            word_set.add(line.rstrip())
    return word_set

def write_word_set_to_file(file_path, word_set):
    with open(file_path, 'w') as f:
        for word in word_set:
            f.write('%s\n' % word)

# Does this support reading/writing unicode characters?
def read_word_translation_dict_from_file(file_path):
    word_translation_dict = {}
    list_complete = False
    with open(file_path, 'r') as f:
        for line in f:
            tokens_found = line.rstrip().split(' ')
            num_tokens = len(tokens_found)
            if num_tokens == 2:
                word_translation_dict[tokens_found[0]] = tokens_found[1]
            elif num_tokens == 1:
                if len(word_translation_dict) == int(tokens_found[0]):
                    list_complete = True
                    break
                else:
                    print("ERROR! Contradicting # at the end of translated pair file!")
                    print( "(%d, %d)" % (len(word_translation_dict), int(tokens_found[0])) )
            else:
                print("ERROR! Incorrect # of tokens found: %d - tokens: %s" % (num_tokens,
                                                                               str(tokens_found)))
    return list_complete, word_translation_dict

def write_word_translation_dict_to_file(file_path, word_translation_dict, is_finished = False):
    with open(file_path, 'w') as f:
        for word, translation in word_translation_dict.items():
            f.write('%s %s\n' % (word, translation))
        if is_finished:
            f.write('%d\n' % len(word_translation_dict))

def pad_dataset_to_equal_length(dataset, max_utterance_len):
    for talk in dataset[0]:
        for utterance in talk:
            utterance += [0] * (max_utterance_len - len(utterance))

def form_datasets(talks, talk_names, test_set_idx, valid_set_idx, train_set_idx):
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
        if talk_names[i] in train_set_idx:
            x_train_list.append( t[0] )
            y_train_list.append( t[1] )
        t_i += 1

    print('Formed dataset appropriately.')
    return ((x_train_list, y_train_list), (x_valid_list, y_valid_list), (x_test_list, y_test_list))

def find_unique_words_in_dataset(talks_read, talk_names, talk_idx, monolingual,
                                 include_idx_set_members = False):
    talk_is_included = lambda c : ( c in talk_idx if include_idx_set_members else c not in talk_idx)

    word_set = set()
    for k, c in enumerate(talks_read):
        if monolingual or talk_is_included(talk_names[k]):
            for u in c[0]:
                for word in u:
                    word_set.add(word.lower())

    return word_set

def add_words_to_word_vec_dict(word_vec_dict, word_set, dictionary, translations = None):
    succeeded_to_find_in_src_list = 0
    failed_to_find_in_src_list = 0
    for word in word_set:
        try:
            translation = word if translations is None else translations[word]
            word_vec_dict[translation] = dictionary[ translation ]
            succeeded_to_find_in_src_list += 1
        except KeyError as e:
            failed_to_find_in_src_list += 1
    assert(len(word_vec_dict) > 0) # Makes sure num_word_dimensions is assigned a value

    print("# src: %d - %d" % (succeeded_to_find_in_src_list, failed_to_find_in_src_list))
    print("source list size: %d" % len(word_set))
    print("word_vec_dict size: %d" % len(word_vec_dict))


def form_word_vec_dict(dataset, talks_read, talk_names, monolingual, src_word_set, target_word_set,
                       translated_word_dict, translated_pairs_file,
                       source_lang_embedding_file, target_lang_embedding_file,
                       source_lang_transformation_file, target_lang_transformation_file,
                       translation_complete):
    if dataset == 'SwDA':
        test_set_idx = swda_test_set_idx
    elif dataset == 'MRDA':
        test_set_idx = mrda_test_set_idx
    else:
        print("Dataset unknown!")
        exit(0)

    if monolingual:
        source_dictionary = FastVector(vector_file=source_lang_embedding_file)
        word_vec_dict = {}
        add_words_to_word_vec_dict(word_vec_dict, src_word_set, source_dictionary)
        print("Formed word dictionary with language vectors.")

        del source_dictionary
        del src_word_set
    else:
        if translated_word_dict is None:
            translated_word_dict = {}
        else:
            total_not_found_words = 0
            for word in src_word_set:
                if word not in translated_word_dict:
                    total_not_found_words += 1
            print("WARNING: %d words not found in translated_word_dict." % total_not_found_words)

        total_words = len(src_word_set)
        total_translated_words = len(translated_word_dict)
        print("Found %d translated word pairs." % total_translated_words)

        target_dictionary = FastVector(vector_file=target_lang_embedding_file)
        print("Target  monolingual language data loaded successfully.")

        if not translation_complete:
            source_dictionary = FastVector(vector_file=source_lang_embedding_file)
            print("Source monolingual language data loaded successfully.")
            source_dictionary.apply_transform(source_lang_transformation_file)
            print("Transformation data applied to source language.")
            target_dictionary.apply_transform(target_lang_transformation_file)
            print("Transformation data applied to target language.")
            print("Translating words seen in dataset:")

            try:
                words_seen = 0
                for word in src_word_set:
                    if word not in translated_word_dict:
                        try:
                            translation = target_dictionary.translate_inverted_softmax(source_dictionary[word],
                                                                                       source_dictionary, 1500,
                                                                                       recalculate=False)
            #                translation = target_dictionary.translate_nearest_neighbor(source_dictionary[word])
                            translated_word_dict[ word ] = translation
                            total_translated_words += 1
                        except KeyError as e:
                            pass
                        words_seen += 1
                    if words_seen % 100 == 0:
                        print("\t- Translated %d out of %d." % (words_seen + total_translated_words, total_words))
            except KeyboardInterrupt as e:
                if translated_pairs_file is not None:
                    write_word_translation_dict_to_file(translated_pairs_file, translated_word_dict)
                sys.exit(0)
            print("Word translation complete.")

            del source_dictionary
            del target_dictionary

            if translated_pairs_file is not None:
                write_word_translation_dict_to_file(translated_pairs_file, translated_word_dict, True)

            print("Source and target dictionaries are deleted.")

            target_dictionary = FastVector(vector_file=target_lang_embedding_file)

        word_vec_dict = {}
        add_words_to_word_vec_dict(word_vec_dict, src_word_set, target_dictionary, translated_word_dict)
        add_words_to_word_vec_dict(word_vec_dict, target_word_set, target_dictionary)
        print("Formed word dictionary with target language vectors.")

        del target_dictionary
        del target_word_set
        del src_word_set

        for k, c in enumerate(talks_read):
            if talk_names[k] not in test_set_idx:
                for u in c[0]:
                    for i, word in enumerate(u):
                        word_lowercase = word.lower()
                        if word_lowercase in translated_word_dict:
                            u[i] = translated_word_dict[word_lowercase]

        del translated_word_dict

    return word_vec_dict
