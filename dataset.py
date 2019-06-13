from swda.swda import CorpusReader
from os import listdir
from os.path import isfile, join
from train_set_preferences import mrda_train_set_idx, mrda_valid_set_idx, mrda_test_set_idx

def load_swda_corpus_data(swda_directory):
    print('Loading SwDA Corpus...')
    corpus_reader = CorpusReader(swda_directory)

    talks = []
    talk_names = []
    tags_seen = set()
    tag_occurances = {}
    for transcript in corpus_reader.iter_transcripts(False):
        name = 'sw' + str(transcript.conversation_no)
        talk_names.append(name)
        conversation_content = []
        conversation_tags = []
        for utterance in transcript.utterances:
            conversation_content.append( utterance.text_words(True) )
            tag = utterance.damsl_act_tag()
            conversation_tags.append( tag )
            if tag not in tags_seen:
                tags_seen.add(tag)
                tag_occurances[tag] = 1
            else:
                tag_occurances[tag] += 1
        talks.append( (conversation_content, conversation_tags) )

    print('\nFound ' + str(len(tags_seen))+ ' different utterance tags.\n')

    tag_indices = {tag:i for i, tag in enumerate(sorted(list(tags_seen)))}

    for talk in talks:
        talk_tags = talk[1]
        for i, tag in enumerate(talk_tags):
            talk_tags[i] = tag_indices[ tag ]

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tag_indices, tag_occurances

def load_mrda_corpus_data(mrda_directory):
    print('Loading MRDA Corpus...')

    talks = []
    talk_names = []
    tags_seen = set()
    tag_occurances = {}

    talk_names_set = mrda_train_set_idx.union(mrda_valid_set_idx).union(mrda_test_set_idx)
    talk_names = list(talk_names_set)
    file_list = [f for f in listdir(mrda_directory) if isfile(join(mrda_directory, f))]
    file_list = list(filter(lambda f: f in talk_names_set, file_list))


    for talk in talk_names:
        utterances = []
        tags = []
        with open(join(mrda_directory, '%s.out' % talk), 'r') as f:
            for line in f:
                line_components = line.split(',')
                utterance_id = line_components[0]
                utter_text = line_components[1]
                utter_speech_act = line_components[3]
                if len(utter_speech_act) == 0:
                    continue
                utterances_to_add = []
                for utterance in utter_text.split('|'):
                    utterances_to_add.append(utterance.split())

                separated_tags = utter_speech_act.split('|')
                if '.' in separated_tags[-1]:
                    last_tag = separated_tags[-1]
                    separated_tags = separated_tags[:-1] + last_tag.split('.', 1)

                tags_to_add = []
                if len(separated_tags) == 1 and separated_tags[0][0] == '.':
                    #Disruption form
                    tags_to_add.append('d')
                elif len(separated_tags) == 1 and separated_tags[0][0] == 'z':
                    #Purposefully untagged utterance
                    for utter in utterances_to_add:
                        tags_to_add.append('z')
                    #continue
                else:
                    for tag in separated_tags:
                        tag_initial = tag[0]
                        if tag_initial == 'h':
                            tag_initial = 'f'
                        elif tag_initial == 'x' or tag_initial == '%':
                            tag_initial = 'd'
                        elif tag_initial != 'f' and tag_initial != 'b' and tag_initial != 'q' and tag_initial != 's':
                            print("PROBLEM:")
                            print("Weird tag found!")
                            print("Talk name: %s" % talk)
                            print("Utterance ID: %s" % utterance_id)
                            print("Tag: %s" % tag_to_use)
                            print("Tag in question: %s" % tag)
                            exit(0)

                        tags_to_add.append(tag_initial)

                for tag in tags_to_add:
                    if tag in tag_occurances:
                        tag_occurances[tag] += 1
                    else:
                        tags_seen.add(tag)
                        tag_occurances[tag] = 1

                if len(utterances_to_add) == len(tags_to_add) - 1 and tags_to_add[-1] == 'd':
                    del tags_to_add[-1]
                elif len(utterances_to_add) != len(tags_to_add):
                    print("PROBLEM A:")
                    print("Tags are not equal to utterances!")
                    print("Talk name: %s" % talk)
                    print("Utterance ID: %s" % utterance_id)
                    print("DA Label: %s" % tag_to_use)
                    print("Tag in question: %s" % tag)
                    print("len(utterance_to_add): %d" % len(utterances_to_add))
                    print("len(tags_to_add): %d" % len(tags_to_add))
                    print("utterance_to_add: %s" % str(utterances_to_add))
                    print("tags_to_add: %s" % str(tags_to_add))

                utterances += utterances_to_add
                tags += tags_to_add

        if len(utterances) != len(tags):
            print("PROBLEM:")
            print("Talk name: %s" % talk)
            print("Utterance: %d" % len(utterances))
            print("Tags: %d" % len(tags))
            exit(0)
        talks.append( (utterances, tags) )

    print('\nFound ' + str(len(tags_seen))+ ' different utterance tags.\n')

    tag_indices = {tag:i for i, tag in enumerate(sorted(list(tags_seen)))}

    for talk in talks:
        talk_tags = talk[1]
        for i, tag in enumerate(talk_tags):
            talk_tags[i] = tag_indices[ tag ]

    print('Loaded MRDA Corpus.')

    return talks, talk_names, tag_indices, tag_occurances
