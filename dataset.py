from swda.swda import CorpusReader

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
        conversation_tags = talk[1]
        for i in range(len(conversation_tags)):
            conversation_tags[i] = tags_seen[ conversation_tags[i] ]

    print('Loaded SwDA Corpus.')
    return talks, talk_names, tags_seen, tag_occurances


