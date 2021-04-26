import re
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import numpy as np

def load_index_term(input_file):
    with open(input_file) as f:
        index_list_term = []
        is_first_time = True
        pattern_term = '(.+):'
        pattern_docID_position = '(.+):(.+)'
        for line in f:
            if line[0] != ' ':
                if is_first_time:
                    term_index_dict = defaultdict(list)
                    is_first_time = False
                else:
                    index_list_term.append(term_index_dict)
                    term_index_dict = defaultdict(list)
                term = re.search(pattern_term, line)
                term = term.group(1)
                term_index_dict[term].append(term)
            else:
                docID_position = line[1:]
                docID_position = re.search(pattern_docID_position, docID_position)
                docID = docID_position.group(1)
                docID = int(docID)
                positions = docID_position.group(2)
                positions = positions.split(',')
                for position in positions:
                    position = int(position.strip())
                    term_index_dict[docID].append(position)
    return index_list_term

def load_index_phrase(input_file):
    with open(input_file) as f:
        phrase_index_dict = defaultdict(set)
        pattern_phrase = '(.+):'
        pattern_docID = '\t(.+)'
        for line in f:
            if re.match(pattern_phrase, line):
                phrase = re.search(pattern_phrase, line)
                phrase = phrase.group(1)
            elif re.match(pattern_docID, line):
                pos = re.search(pattern_docID, line)
                pos = pos.group(1)
                phrase_index_dict[phrase].add(pos)
    return phrase_index_dict

def process_query(query):
    query = re.sub(r'[^A-Za-z\s]', '', query)
    query = query.lower()
    querysplit = query.split()
    querysplit = [word.strip() for word in querysplit]
    querysplit = [word for word in querysplit if not word in stword_set]
    querystemmer = [SnowballStemmer('english').stem(word) for word in querysplit]
    return querystemmer

def search(input_query):
    pattern_query_ID_Question = '(\d+) (.+)'
    query_ID_Question = re.search(pattern_query_ID_Question, input_query)
    queryID = query_ID_Question.group(1)
    queryID = queryID.strip()    
    pattern_AND = '(.+)AND(.+)'
    pattern_PHRASE = '"(.+) (.+)"'
    pattern_PROX = '#(\d+)\((.+),(.+)\)'
    if re.match(pattern_AND, query_ID_Question.group(2)): #as a whole the input_query has AND operator 
        s = AND_search(pattern_AND, query_ID_Question.group(2))        
    elif re.match(pattern_PHRASE, query_ID_Question.group(2)):#as a whole the input_query is a phrase
        s = PHRASE_search(pattern_PHRASE, query_ID_Question.group(2))
    elif re.match(pattern_PROX, query_ID_Question.group(2)):#as a whole the input_query is searching distance
        s = PROX_search(pattern_PROX, query_ID_Question.group(2))
    else:
        s = single_word_search(query_ID_Question.group(2))
    write_to_file(queryID, s)

def write_to_file(queryID, s):
    with open('results.boolean.txt', 'a') as f:
        for i in s:
            f.write(str(queryID) + ' 0 ' + str(i) + ' 0 1 0 \n')


def PHRASE_search(pattern_PHRASE, query):
    query = re.search(pattern_PHRASE, query)
    q1 = query.group(1)
    q1 = process_query(q1)[0]
    q2 = query.group(2)
    q2 = process_query(q2)[0]
    q = q1 + '_' + q2
    s = phrase_get_result_set(q, index_list_phrase)
    return s

def single_word_search(query):
    q = process_query(query)[0]
    s = get_result_set(q, index_list_term)
    return s

def NOT_search(pattern_NOT, query):
    query = re.search(pattern_NOT, query)    
    q = query.group(1)
    q = process_query(q)[0]
    s = get_result_set(q, index_list_term)
    return s

def AND_search(pattern_AND, query):
    query = re.search(pattern_AND, query)
    q1 = query.group(1)
    q1 = process_query(q1)[0]
    q2 = query.group(2)
    q2 = process_query(q2)[0]
    result = []
    pattern_NOT = 'NOT (.+)'
    pattern_PHRASE = '"(.+) (.+)"'
    AND_query_list = [q1,q2]
    for q in AND_query_list:        
        if re.match(pattern_PHRASE, q): #judge whether q is a phrase
            phrase_set = PHRASE_search(pattern_PHRASE, q)
            result.append(1) #1 indicates that the set after 1 is used to plus
            result.append(phrase_set)
        elif re.match(pattern_NOT, q): #judge whether q is a not query
            not_set = NOT_search(pattern_PHRASE, q)            
            result.append(-1) #-1 indicates that the set after -1 is used to minus
            result.append(not_set)
        else:
            single_word_set = single_word_search(q)
            result.append(1)
            result.append(single_word_set)
    q_set1 = result[1]
    q_set2 = result[3]
    if result[0] < 0:
        s = q_set2.difference(q_set1)
    elif result[2] < 0:
        s = q_set2.difference(q_set1)
    else:
        s = q_set1.intersection(q_set2)
    return s

def PROX_search(pattern_PROX, query):
    s = set()
    query = re.search(pattern_PROX, query)
    q1 = query.group(1)#get distence
    q1 = int(q1)
    q2 = query.group(2)
    q2 = process_query(q2)[0]
    q3 = query.group(3)
    q3 = process_query(q3)[0]
    q2_set = get_result_set(q2, index_list_term)
    q3_set = get_result_set(q3, index_list_term)
    common_set = q2_set.intersection(q3_set)#find documents both q2 and q3 appeared
    for t2 in index_list_term:
        if q2 in t2.keys():
            q2_dict = t2       #get q2 information from index, q2_dict={q2:[q2],docID:[pos1,pos2,...],...}
            break
    for t3 in index_list_term:
        if q3 in t3.keys():
            q3_dict = t3
            break
    is_find = False
    for c_set in common_set:
        q2_positions = q2_dict[c_set]
        q3_positions = q3_dict[c_set]
        for i in q2_positions:
            for j in q3_positions:
                distance = abs(i-j)  #compute any possible distance
                if distance - 1 <= q1: #if there exists a distence smaller than q1
                    is_find = True
                    s.add(c_set)
                    break
            if is_find:
                break
    return s

def get_result_set(q, index_list):
    q_set = set()
    for t in index_list:
        if q in t.keys():
            key_list = list(t.keys())[1:]
            for k in key_list:
                q_set.add(k)
            break
    return q_set

def phrase_get_result_set(q, index_dict):
    q_set = set()
    if q in index_dict.keys():
        q_set = index_dict[q]
    return q_set

def boolean_search(search_file):
    with open(search_file) as f:
        for line in f:
            search(line)

def load_stop_word(input_stword):
    with open(input_stword) as f:
        stword_set = set()
        for stword in f:
            stword = stword.strip()
            stword_set.add(stword)
    return stword_set

def document_number(input_file):
    with open(input_file, encoding = 'gb18030', errors = 'ignore') as f:
        pattren_docID = '(ID: )(\d+)'
        document_number = 0
        documentID_list = []
        for line in f:
            if re.match( pattren_docID, line):
                docIDline = re.search(pattren_docID, line)
                docID = int(docIDline.group(2))
                documentID_list.append(docID)
                document_number = document_number + 1
    return document_number, documentID_list

def compute_tf(index_list_term):
    tf_list = []    
    for index_dict_term in index_list_term:
        tf_dict = defaultdict(int)
        keys_list = list(index_dict_term.keys())
        term = keys_list[0]
        tf_dict[term] = 0        
        for docID in keys_list[1:]:            
            document = docID            
            frequency = len(index_dict_term[docID])            
            tf_dict[document] = frequency
        tf_list.append(tf_dict)
    return tf_list

def compute_df(index_list_term):
    df_dict = defaultdict(int)
    for index_dict_term in index_list_term:
        keys_list = list(index_dict_term.keys())
        term = keys_list[0]
        document_frequency = len(keys_list[1:])
        df_dict[term] = document_frequency
    return df_dict

def compute_idf():
    df_dict = compute_df(index_list_term)
    idf_dict = defaultdict(float)
    for term in df_dict.keys():
        idf = np.log10(document_number/df_dict[term])
        idf_dict[term] = idf
    return idf_dict

def compute_weight():
    tf_list = compute_tf(index_list_term)
    idf_dict = compute_idf()
    weight_list = []
    for tf_dict in tf_list:
        weight_dict = defaultdict(float)
        keys_list = list(tf_dict.keys())
        term = keys_list[0]
        weight_dict[term] = 0
        idf = idf_dict[term]
        for document in keys_list[1:]:
            weight = (1 + np.log10(tf_dict[document])) * idf
            weight_dict[document] = weight
        weight_list.append(weight_dict)
    return weight_list

def compute_rank(query):
    querystemmer = process_query(query)
    weight_list = compute_weight()    
    rank_result_dict = defaultdict(float)
    for stemmer in querystemmer:
        for weight_dict in weight_list:
            if stemmer in weight_dict:
                keys_list = list(weight_dict.keys())
                for document in keys_list[1:]:
                    rank_result_dict[document] = rank_result_dict[document] + weight_dict[document]
    for d in documentID_list: #because some document didn't exist in rank_result_dict, we need to complete it
        if d in rank_result_dict.keys():
            pass
        else:
            rank_result_dict[d] = 0
    rank_result_list_sorted = sorted(rank_result_dict.items(), key = lambda item:item[1], reverse=True)    
    return rank_result_list_sorted

def search_rank(input_query):
    pattern_query_ID_Question = '(\d+) (.+)'
    query_ID_Question = re.search(pattern_query_ID_Question, input_query)#get queryID and query content
    queryID = query_ID_Question.group(1)
    queryID = queryID.strip()
    rank_result_list_sorted = compute_rank(query_ID_Question.group(2))
    write_file(queryID, rank_result_list_sorted)
    
def write_file(queryID, result):
    with open('results.ranked.txt', 'a') as f:
        for i in result[:1000]:
            f.write(str(queryID) + ' 0 ' + str(i[0]) + ' 0 ' + str(i[1]) + ' 0 \n')

def rank_search(search_file):
    with open(search_file) as f:
        for line in f:
            search_rank(line)


index_list_term = load_index_term('index.txt')
index_list_phrase = load_index_phrase('output_index_phrase.txt')
stword_set = load_stop_word('englishST.txt')
boolean_search('queries.boolean.txt')
document_number,documentID_list = document_number('trec.5000.txt')
rank_search('queries.ranked.txt')










