import spacy
import numpy as np

MAX_SENTENCE_LENGTH = 30
def get_question_features(question):
    ''' For a given question, returns a vector with
    each word (token) transformed into a 300 dimension 
    representation calculated using Glove '''

    # en_core_web_md may be better?
    processor = spacy.load('en_vectors_glove_md')
    tokens = processor(question)
    question_tensor = np.zeros((1, MAX_SENTENCE_LENGTH, 300))
    for i in range(len(tokens)):
        question_tensor[0, i, :] = tokens[i].vector

    return question_tensor

# usage
# remember to use unicode string, e.g. unicode(str, 'utf-8')
bob = get_question_features(u'Everything is gonna be alright')
print(bob.shape)
