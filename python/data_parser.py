import xml.etree.ElementTree

e = xml.etree.ElementTree.parse('../data/kpwr-1.2.7-names-disamb-nam-flatten/techniczne/00101534.xml').getroot()
# e = xml.etree.ElementTree.parse('../data/kpwr-1.2.7-names-disamb-nam-flatten/stenogramy/00101522.xml').getroot()

printType = True

for chunk in e.findall('chunk'):
    for sentence in chunk.findall('sentence'):
        properName = ""
        nameId = '0'
        for tok in sentence.findall('tok'):
            for ann in tok.findall('ann'):
                if ann.attrib['chan'] == 'nam' and ann.text != nameId and properName != "":
                    if printType:
                        properName += " " + nameId
                    print properName
                    properName = ""
                if ann.attrib['chan'] == 'nam' and ann.text != '0':
                    nameId = ann.text
                    if properName != "":
                        properName += " "
                    properName += tok.find('orth').text
        if properName != "":
            if printType:
                properName += " " + nameId
            print properName
