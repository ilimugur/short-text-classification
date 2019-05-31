from swda.swda import CorpusReader
from os import listdir
from os.path import isfile, join

def load_swda_corpus_data(swda_directory):
    print('Loading SwDA Corpus...')
    corpus_reader = CorpusReader(swda_directory)

    talks = []
    talk_names = []
    tags_seen = {}
    tag_occurances = {}
    num_tags_seen = 0
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
                tags_seen[tag] = num_tags_seen
                num_tags_seen += 1
                tag_occurances[tag] = 1
            else:
                tag_occurances[tag] += 1
        talks.append( (conversation_content, conversation_tags) )

    print('\nFound ' + str(len(tags_seen))+ ' different utterance tags.\n')

    for talk in talks:
        talk_tags = talk[1]
        for i, tag in enumerate(talk_tags):
            talk_tags[i] = tags_seen[ tag ]

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tags_seen, tag_occurances

def load_mrda_corpus_data(mrda_directory):
    print('Loading MRDA Corpus...')

    talks = []
    talk_names = []
    tags_seen = {}
    tag_occurances = {}
    num_tags_seen = 0

    file_list = [f for f in listdir(mrda_directory) if isfile(join(mrda_directory, f))]
    talk_names_set = set(f.split('.')[0] for f in file_list)
    talk_names = list(talk_names_set)
    print(talk_names)
    for talk in talk_names:
        # Get utterances
        utterance_map = {}
        with open(join(mrda_directory, '%s.trans' % talk), 'r') as f:
            for line in f:
                line_components = line.split(',')
                utterance_id = line_components[0]
                utterance = line_components[1]
                separated_utterances = [utterance.split() for utterance in utterance.split('|')]
                utterance_map[utterance_id] = separated_utterances

        # Get tag
        tags = []
        utterances = []
        with open(join(mrda_directory, '%s.dadb' % talk), 'r') as f:
            for line in f:
                utterance_data_components = line.split(',')
                utterance_id = utterance_data_components[2]
                da_label = utterance_data_components[5]

                tag_to_use = da_label
                if len(tag_to_use) == 0:
                    continue

                separated_tags = tag_to_use.split('|')
                if '.' in separated_tags[-1]:
                    last_tag = separated_tags[-1]
                    separated_tags = separated_tags[:-1] + last_tag.split('.', 1)

                tags_to_add = []
                if len(separated_tags) == 1 and separated_tags[0][0] == '.':
                    #Disruption form
                    tags_to_add.append('d')
                elif len(separated_tags) == 1 and separated_tags[0][0] == 'z':
                    #Purposefully untagged utterance
                    continue
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

                        if tag_initial not in tags_seen:
                            tags_seen[tag_initial] = num_tags_seen
                            num_tags_seen += 1
                            tag_occurances[tag_initial] = 1
                        else:
                            tag_occurances[tag_initial] += 1

                        tags_to_add.append(tag_initial)

                utterance_to_add = utterance_map[utterance_id]

                if len(utterance_to_add) == len(tags_to_add) - 1 and tags_to_add[-1] == 'd':
                    del tags_to_add[-1]
                elif len(utterance_to_add) != len(tags_to_add):
                    print("PROBLEM:")
                    print("Tags are not equal to utterances!")
                    print("Talk name: %s" % talk)
                    print("Utterance ID: %s" % utterance_id)
                    print("DA Label: %s" % tag_to_use)
                    print("Tag in question: %s" % tag)
                    print("len(utterance_to_add): %d" % len(utterance_to_add))
                    print("len(tags_to_add): %d" % len(tags_to_add))
                    print("utterance_to_add: %s" % str(utterance_to_add))
                    print("tags_to_add: %s" % str(tags_to_add))
                    exit(0)
                tags += tags_to_add
                utterances += utterance_to_add
#                print("%s\t%s" % (str(tags_to_add), str(utterance_to_add)))

        if len(utterances) != len(tags):
            print("PROBLEM:")
            print("Talk name: %s" % talk)
            print("Utterance: %d" % len(utterances))
            print("Tags: %d" % len(tags))
            exit(0)
        talks.append( (utterances, tags) )

    print('\nFound ' + str(len(tags_seen))+ ' different utterance tags.\n')

    for talk in talks:
        talk_tags = talk[1]
        for i, tag in enumerate(talk_tags):
            talk_tags[i] = tags_seen[ tag ]

    print('Loaded MRDA Corpus.')
    return talks, talk_names, tags_seen, tag_occurances
