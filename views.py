# -*- coding: utf-8 -*-

"""
This script obtains the replacement pair, along with the
position and length of the input word in the input sentence.
Additionally, every other information for the JSON output is
defined.
"""

from gectorPredict.predict import predict_for_paragraph
from gectorPredict.gector.gec_model import GecBERTModel

args = {'vocab_path':'gectorPredict/MODEL_DIR/vocabulary/', 'model_path':['gectorPredict/MODEL_DIR/best.th'],
        'max_len':50,'min_len':3,'iteration_count':5,'min_error_probability':0.7,
        'lowercase_tokens':0,'transformer_model':'bertimbaubase','special_tokens_fix':1,'additional_confidence':0.3,
        'is_ensemble':0,'weights':None}

model = GecBERTModel(model_paths=args['model_path'],
                     vocab_path=args['vocab_path'],
                     max_len=args['max_len'], min_len=args['min_len'],
                     iterations=args['iteration_count'],
                     min_error_probability=args['min_error_probability'],
                     lowercase_tokens=args['lowercase_tokens'],
                     model_name=args['transformer_model'],
                     special_tokens_fix=args['special_tokens_fix'],
                     log=False,
                     confidence=args['additional_confidence'],
                     is_ensemble=args['is_ensemble'],
                     weigths=args['weights'])

#%%

request_string = 'Poderá haver muitas pessoas na praia.'
repl = predict_for_paragraph(request_string, model, tokenizer_method='spacy')



json_output = dict()
json_output['software'] = {'deep3SPVersion':'0.6'}
json_output['warnings'] = {'incompleteResults':False}
json_output['language'] = {'name':'Portuguese (Deep SymFree)'}
json_output['matches'] = []
for i, (key, value) in enumerate(zip(repl.keys(), repl.values())):
    match_dict = dict()
    match_dict['message'] = 'O verbo <marker>' + request_string[value[0]:value[0]+value[1]] + '</marker> não concorda com o resto da frase ou não é frequentemente utilizado neste contexto. Considere a alternativa.'
    match_dict['incorrectExample'] = 'As pessoas foi para casa.'
    match_dict['correctExample'] = 'As pessoas foram para casa.'
    match_dict['shortMessage'] = 'Modifique a forma verbal'
    match_dict['replacements'] = [{'value':value[2]}]
    match_dict['offset'] = value[0]
    match_dict['length'] = value[1]
    match_dict['context'] = {'text':request_string, 'offset':value[0], 'length':value[1]}
    match_dict['sentence'] = request_string
    match_dict['type'] = {'typeName':'Hint'}
    match_dict['rule'] = {'id':'DEEP_VERB_3SP', 'subId':0, 'sourceFile': 'not well defined', 'description': 'Deep learning rules for the 3rd person Singular-Plural', 'issueType':'grammar', 'category':{'id':'SymFree_DEEP_1' , 'name':'Deep learning rules (SymFree 1)'}}
    match_dict['ignoreForIncompleteSentence'] = False
    match_dict['contextForSureMatch'] = -1
    json_output['matches'].append(match_dict)

json_output











