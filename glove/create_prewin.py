import os
import sys
import codecs
from spacy.lang.en import English
import spacy
import numpy as np

from stats import dump_to_pickle

path = sys.argv[1]
# path = "~/metonymy-resolution/harvest-data/disambiguation-pages/corpora/new-corpora/prewin-multi/wiki_LOCATION_train.txt"  # Input file name.
dirname = os.path.dirname(path)
name = os.path.basename(path)
rawname = os.path.splitext(name)[0] # without extension

if 'lit' in name or 'literal' in name or 'LOCATION' in name:
    label = 0
else:
    if 'met' in name or 'metonymic' in name or 'mixed' in name:
        label = 1 # 1 is for METONYMY/NON-LITERAL, 0 is for LITERAL
    elif 'INSTITUTE' in name:
        label = 1
    elif 'TEAM' in name:
        label = 2
    elif 'ARTIFACT' in name:
        label = 3
    elif 'EVENT' in name:
        label = 4

tokenizer = English(parser=False)
en_nlp = spacy.load('en')
inp = codecs.open(path, mode="r", encoding="utf-8")
# PLEASE FORMAT THE INPUT FILE AS ONE SENTENCE PER LINE. SEE BELOW:
# ENTITY<SEP>sentence<ENT>ENTITY<ENT>rest of sentence.
# Germany<SEP>Their privileges as permanent Security Council members, especially the right of veto, 
# had been increasingly questioned by <ENT>Germany<ENT> and Japan which, as major economic powers.
out = []
seq_length = 5  # A window of 5 is the DEFAULT for the PUBLICATION methodology. Feel free to experiment.


def locate_entity(document, ent, left_w, right_w):
    left_w = '' if len(left_w) == 0 else left_w[-1].text
    right_w = '' if len(right_w) == 0 else right_w[0].text
    for doc in document:
        if doc.text == ent[0]:
            index = doc.i
            if left_w == '' or document[index - 1].text == left_w:
                if right_w == '' or document[index + len(ent)].text == right_w:
                    return index + len(ent) - 1
    raise Exception()  # If this is ever triggered, there are problems parsing the text. Check SpaCy output!


def find_start(old):
    while old.dep_ == "conj":
        old = old.head
    return old.head


def pad(coll, from_left):
    while len(coll) < seq_length:
        if from_left:
            coll = [u"0.0"] + coll
        else:
            coll += [u"0.0"]
    return coll


for line in inp:
    line = line.split(u"<SEP>")
    sentence = line[1].split(u"<ENT>")
    entity = [t.text for t in tokenizer(sentence[1])]
    en_doc = en_nlp(u"".join(sentence).strip())
    left = seq_length * [u"0.0"]
    right = seq_length * [u"0.0"]
    dep_left = seq_length * [u"0.0"]
    dep_right = seq_length * [u"0.0"]
    words = []
    index = locate_entity(en_doc, entity, tokenizer(sentence[0].strip()), tokenizer(sentence[2].strip()))
    start = find_start(en_doc[index])
    if start.i > index:
        if index + 1 < len(en_doc) and en_doc[index + 1].dep_ in [u"case", u"compound", u"amod"] \
                and en_doc[index + 1].head == en_doc[index]:  # any neighbouring word that links to it
            right = pad([en_doc[index + 1].text] + [t.text for t in en_doc[start.i:][:seq_length - 1]], False)
            dep_right = pad([en_doc[index + 1].dep_] + [t.dep_ for t in en_doc[start.i:]][:seq_length - 1], False)
        else:
            right = pad([t.text for t in en_doc[start.i:][:seq_length]], False)
            dep_right = pad([t.dep_ for t in en_doc[start.i:]][:seq_length], False)
    else:
        if index - len(entity) >= 0 and en_doc[index - len(entity)].dep_ in [u"case", u"compound", u"amod"] \
                and en_doc[index - len(entity)].head == en_doc[index]:  # any neighbouring word that links to it
            left = pad([t.text for t in en_doc[:start.i + 1][-(seq_length - 1):]] + [en_doc[index - len(entity)].text], True)
            dep_left = pad([t.dep_ for t in en_doc[:start.i + 1]][-(seq_length - 1):] + [en_doc[index - len(entity)].dep_], True)
        else:
            left = pad([t.text for t in en_doc[:start.i + 1][-seq_length:]], True)
            dep_left = pad([t.dep_ for t in en_doc[:start.i + 1]][-seq_length:], True)
    out.append((left, dep_left, right, dep_right, label))
    # print(left, right)
    # print(dep_left, dep_right)
    # print(label)
    # print(line[1])
print("Processed:{} lines/sentences.".format(len(out)))
dump_to_pickle("{}/pickle/{}.pkl".format(dirname, rawname), out)
