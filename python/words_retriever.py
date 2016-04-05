# coding=utf-8
import xml.etree.ElementTree


def get_words(file_path):
    with open(file_path, 'r') as xml_file:
        e = xml.etree.ElementTree.parse(xml_file).getroot()

        inner_dictionary = {}
        for chunk in e.findall('chunk'):
            for sentence in chunk.findall('sentence'):
                for tok in sentence.findall('tok'):
                    is_proper_name = False
                    for ann in tok.findall('ann'):
                        if ann.attrib['chan'] == 'nam' and ann.text != '0':
                            is_proper_name = True
                    for orth in tok.find('orth'):
                        inner_dictionary[orth.text] = is_proper_name
                    for lex in tok.findall('lex'):
                        inner_dictionary[lex.find('base').text] = is_proper_name
        return inner_dictionary


with open('../data/kpwr-1.2.7-names-disamb-nam-flatten/index_names.txt', 'rU') as f:
    file_ = open('../data/prepared/AllFlattenNamWords.txt', 'w')
    file_.truncate()
    for line in f:
        # try:
            dictionary = get_words("../data/kpwr-1.2.7-names-disamb-nam-flatten/"+line.decode("utf-8").rstrip('\n'))
            for word in dictionary:
                if dictionary[word]:
                    file_.write((word + " 1\n").encode("utf-8"))
                else:
                    file_.write((word + " 0\n").encode("utf-8"))
        # except UnicodeDecodeError:
        #     print line
    file_.close()
