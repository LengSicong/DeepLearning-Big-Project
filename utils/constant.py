"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids

# NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,"OTHER": 2, "CARDINAL": 3, "GPE": 4, "DATE": 5, "PERSON": 6, "ORDINAL": 7, "NORP": 8, "EVENT": 9, "PRODUCT": 10, "ORG": 11, "MONEY": 12, "QUANTITY": 13, "FAC": 14, "TIME": 15, "LOC": 16, "WORK_OF_ART": 17, "LANGUAGE": 18, "PERCENT": 19, "LAW": 20}

# POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,"DT": 2, "JJ": 3, "NN": 4, "VBZ": 5, "VBN": 6, "VBG": 7, "IN": 8, "CC": 9, "PRP$": 10, "_SP": 11, "PRP": 12, "VBP": 13, "NNS": 14, "NNP": 15, "TO": 16, "VB": 17, "CD": 18, "RB": 19, "RP": 20, "FW": 21, "ADD": 22, "EX": 23, "VBD": 24, "WRB": 25, "JJR": 26, "WDT": 27, "JJS": 28, "WP": 29, ".": 30, "RBR": 31, "MD": 32, ":": 33, "''": 34, "NNPS": 35, "XX": 36, "UH": 37, "-LRB-": 38, ",": 39, "``": 40, "PDT": 41, "HYPH": 42, "WP$": 43, "-RRB-": 44, "LS": 45, "NFP": 46, "AFX": 47, "RBS": 48, "SYM": 49, "$": 50, "POS": 51}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}



INFINITY_NUMBER = 1e12