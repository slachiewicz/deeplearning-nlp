#!/usr/bin/python
# coding=utf-8
import xml.etree.ElementTree
import re

def get_sentences(file_path):
    with open(file_path, 'r') as xml_file:
        e = xml.etree.ElementTree.parse(xml_file).getroot()

        sentences = []
        for chunk in e.findall('chunk'):
            for sentence in chunk.findall('sentence'):
                list_of_words, list_of_values = create_data(sentence)
                combined = create_output_line(list_of_words, list_of_values) 
                sentences.append(combined)
        return sentences

def create_data(xml_sentence):
    list_of_words = []
    list_of_values = []
    for tok in xml_sentence.findall('tok'):
        orth = tok.find('orth').text
        if re.search("^[.(){},?\";-]+", orth) == None:
            list_of_words.append(orth)
            list_of_values.append(generate_value_for_token(tok))
    return list_of_words, list_of_values

def generate_value_for_token(token):
    for ann in token.findall('ann'):
        if ann.attrib['chan'] == 'nam' and ann.text != '0':
            return '1'
    return '0'

def create_output_line(list_of_words, list_of_values):
    sentence = u" ".join(list_of_words).encode("utf-8")
    values = u";".join(list_of_values).encode("utf-8")
    combined = ",".join([sentence, values]) + '\n'
    return combined

with open('../data/kpwr-1.2.7-names-disamb-nam-flatten/index_names.txt', 'rU') as f:
    file_ = open('../data/prepared/AllFlattenNamSentences.csv', 'w')
    file_.truncate()
    for line in f:
        file_name = "../data/kpwr-1.2.7-names-disamb-nam-flatten/" + line.decode("utf-8").rstrip('\n')
        all_sentences = get_sentences(file_name)
        file_.writelines(all_sentences)
        print u"File {} done.".format(file_name)
    file_.close()
