from train_set_preferences import mrda_test_set_idx, swda_test_set_idx
from google.cloud import translate
from dataset import load_swda_corpus_data


def translate_test_data_by_words(talks, talk_names, talks_to_translate, language):
    translate_client = translate.Client()
    numChars = 0
    for i in range(len(talks)):
        if talk_names[i] in talks_to_translate:
            conversation, _ = talks[i]
            print("Translating talk: %s" % talk_names[i])
            for j in range(len(conversation)):
                utterance = conversation[j]
                for k in range(len(utterance)):
                    word = utterance[k]
                    numChars += len(word)
                    translation = translate_client.translate(word, target_language=language)
                    utterance[k] = translation['translatedText']
            print("Translated talk: %s" % talk_names[i])

    print("Total characters translated=" + str(numChars))

    return talks, talk_names

def translate_test_data_by_utterances(talks, talk_names, talks_to_translate, language):
    translate_client = translate.Client()

    numChars = 0
    for i in range(len(talks)):
        if talk_names[i] in talks_to_translate:
            conversation, _ = talks[i]
            print("Translating talk: %s" % talk_names[i])
            for j in range(len(conversation)):
                utterance = " ".join(conversation[j])
                numChars += len(utterance)
                translation = translate_client.translate(utterance, target_language=language)
                translated_utterance = translation['translatedText']
                conversation[j] = translated_utterance.split()
            print("Translated talk: %s" % talk_names[i])
                

    print("Total characters translated=" + str(numChars))

    return talks, talk_names

def translate_and_store_swda_corpus_test_data(dataset, dataset_loading_function, dataset_file_path, translation_file_path, language, translate_whole_utterances = True):
    talks_read, talk_names, _, _ = dataset_loading_function(dataset_file_path)

    if dataset == 'MRDA':
        test_set_idx = mrda_test_set_idx
    elif dataset == 'SwDA':
        test_set_idx = swda_test_set_idx
    else:
        print("Unknown dataset!")
        exit(0)

    if translate_whole_utterances:
        unit_str = 'u'
    else:
        unit_str = 'w'

    print( "# of talks read:" + str( len(talks_read) ) )

    for i in range(len(talks_read)):
        if talk_names[i] in test_set_idx:
            if translate_whole_utterances:
                talk_read, talk_name = translate_test_data_by_utterances([talks_read[i]], [talk_names[i]], test_set_idx, language)
            else:
                talk_read, talk_name = translate_test_data_by_words([talks_read[i]], [talk_names[i]], test_set_idx, language)
            talk_read = talk_read[0]
            talk_name = talk_name[0]

            print("Storing file: %s" % talk_names[i])
            fileName = translation_file_path + talk_name + "_" + language + "_" + unit_str + '.txt'
            print(fileName)
            f = open(fileName, 'w')
            conversation = talk_read[0]
            f.write(str(len(conversation)) + '\n')
            for utterance in conversation:
                f.write(str(len(utterance)) + '\n')
                utterance_string = " ".join(utterance)
                f.write(utterance_string + '\n')
            tags = talks_read[i][1]
            for tag in tags:
                #print("tag:" + str(tag))
                f.write(str(tag) + '\n')
            f.close()

def read_translated_swda_corpus_data(dataset, talks_read, talk_names, translation_file_path, language, use_utterance_translation = True):
    if use_utterance_translation:
        unit_str = 'u'
    else:
        unit_str = 'w'

    if dataset == 'MRDA':
        test_set_idx = mrda_test_set_idx
    elif dataset == 'SwDA':
        test_set_idx = swda_test_set_idx
    else:
        print("Unknown dataset!")
        exit(0)

    for i in range(len(talks_read)):
        if talk_names[i] in test_set_idx:
            f = open(translation_file_path + talk_names[i] + "_" + language + "_" + unit_str + '.txt', 'r')
            conversation = []
            num_utterances = int(f.readline())
            for j in range(num_utterances):
                num_words = int(f.readline())
                utterance_string = f.readline()
                utterance = utterance_string.split()
                conversation.append(utterance)

            tags = []
            for j in range(num_utterances):
                tag = int(f.readline())
                tags.append(tag)
            f.close()

            talks_read[i] = (conversation, tags)

    return talks_read, talk_names
