import xml.etree.ElementTree

e = xml.etree.ElementTree.parse('../data/kpwr-1.2.7-names-disamb-nam-flatten/techniczne/00101534.xml').getroot()

for chunk in e.findall('chunk'):
    for sentence in chunk.findall('sentence'):
        text = ""
        for tok in sentence.findall('tok'):
            for ann in tok.findall('ann'):
                if ann.attrib['chan'] == 'nam' and ann.text != '0':
                    text = text + tok.find('orth').text + " "
        if text != "":
            print text + "\n"
