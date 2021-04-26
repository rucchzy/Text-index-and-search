import re
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict


def preprocess_file(input_file,input_stword):
    with open(input_file, encoding = 'gb18030', errors = 'ignore') as f1:
        with open(input_stword) as f2:
            stword_set = set()          #a set to store stop-words
            for stword in f2:           #generate stop-words set
                stword = stword.strip()
                stword_set.add(stword)
            term_set = set()            #a set to store different terms
            phrase_set = set()          #a set to store different terms
            term_position_list = []     #a list to store different terms position in one document
            phrase_position_dict = defaultdict(list)  #generate a dictionary to store document ID where phrase appeared
            pattren_docID = '(ID: )(\d+)'             
            pattern_headline = '(HEADLINE: )(.+)'
            for line in f1:
                term_position = 1
                if re.match( pattren_docID, line):  #get document ID
                    docIDline = re.search(pattren_docID, line)
                    docID = int(docIDline.group(2))
                elif re.match( pattern_headline, line): #get document headline
                    headline_line = re.search(pattern_headline, line)
                    headline_text = headline_line.group(2)                        
                elif re.match(r'TEXT: ', line):  #get document text
                    text = headline_text + re.sub(r'TEXT: ', '', line) #headline is added to text
                    text = re.sub(r'[^A-Za-z\s]', ' ', text)
                    text = text.lower()
                    textsplit = text.split()
                    textsplit = [word for word in textsplit if not word in stword_set] #remove stop-words
                    textstemmer = [SnowballStemmer('english').stem(word) for word in textsplit]
                    term_position_dict = defaultdict(list)     #generate a dictionary to store positions that a term appeared in a document
                    term_position_dict['docID'].append(docID)  #the first key and value is set as docID
                    for i in range(len(textstemmer)-1):
                        term_position_dict[textstemmer[i]].append(term_position)  #add new (key,value) into dictionary
                        term_position = term_position + 1
                        term_set.add(textstemmer[i])
                        phrase = str(textstemmer[i]) + '_' + str(textstemmer[i+1]) #generate bigram phrase
                        phrase_position_dict[phrase].append(docID)              #add new (key,value) into dictionary
                        phrase_set.add(phrase)
                    term_position_list.append(term_position_dict)               
                else:
                    pass
    return stword_set,term_set,term_position_list,phrase_position_dict

def output_index_term(output_file):
    with open( output_file, 'w') as f:
        for term in term_set:
            f.write(str(term) + ':\n')
            for term_position in term_position_list:
                if term in term_position.keys():
                    f.write('   ' + str(term_position['docID'][0]) + ': ')
                    poslist = term_position[term]
                    newposlist = []
                    for pl in poslist:
                        pl = str(pl)
                        newposlist.append(pl)        
                    p_by_term = ', '.join(newposlist)
                    f.write(p_by_term)               
                    f.write('\n')

def output_index_phrase(output_file):
    with open(output_file, 'w') as f:
        for phrase in phrase_position_dict.keys():
            f.write(phrase + ':\n')
            for docID in phrase_position_dict[phrase]:
                f.write('\t' + str(docID) + '\n')


stword_set,term_set,term_position_list,phrase_position_dict = preprocess_file('trec.5000.txt','englishST.txt')
output_index_term('index.txt')
output_index_phrase('output_index_phrase.txt')