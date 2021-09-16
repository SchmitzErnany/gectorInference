# -*- coding: utf-8 -*-
#%%
"""
This script obtains the replacement pair, along with the
position and length of the input word in the input sentence.
Additionally, every other information for the JSON output is
defined.
"""

from gectorPredict.predict import (
    predict_for_paragraph,
    message,
    short_message,
    examples,
)
from gectorPredict.gector.gec_model import GecBERTModel

args = {
    "vocab_path": "gectorPredict/MODEL_DIR/vocabulary/",
    "model_path": ["gectorPredict/MODEL_DIR/best.th"],
    "max_len": 50,
    "min_len": 2,
    "iteration_count": 5,
    "min_error_probability": 0.7,
    "lowercase_tokens": 0,
    "transformer_model": "bertimbaubase",
    "special_tokens_fix": 1,
    "additional_confidence": 0.3,
    "is_ensemble": 0,
    "weights": None,
}

model = GecBERTModel(
    model_paths=args["model_path"],
    vocab_path=args["vocab_path"],
    max_len=args["max_len"],
    min_len=args["min_len"],
    iterations=args["iteration_count"],
    min_error_probability=args["min_error_probability"],
    lowercase_tokens=args["lowercase_tokens"],
    model_name=args["transformer_model"],
    special_tokens_fix=args["special_tokens_fix"],
    log=False,
    confidence=args["additional_confidence"],
    is_ensemble=args["is_ensemble"],
    weigths=args["weights"],
)

#%%

request_string = "Hoje farei isso."
repl = predict_for_paragraph(request_string, model, tokenizer_method="split+spacy")


json_output = dict()
json_output["software"] = {"deep3SPVersion": "1.0"}
json_output["warnings"] = {"incompleteResults": False}
json_output["language"] = {"name": "Portuguese (Deep SymFree)"}
json_output["matches"] = []
for key, value in zip(repl.keys(), repl.values()):
    original_token = key[0]
    replacement = value["replacement"]
    offset = value["word_position"]
    length = value["word_length"]
    append_id = value["transformation_label"]
    match_dict = dict()
    match_dict["message"] = message(original_token, replacement)
    match_dict["incorrectExample"] = examples(original_token, replacement)[0]
    match_dict["correctExample"] = examples(original_token, replacement)[1]
    match_dict["shortMessage"] = short_message(original_token, replacement)
    match_dict["replacements"] = [{"value": replacement}]
    match_dict["offset"] = offset
    match_dict["length"] = length
    match_dict["context"] = {"text": request_string, "offset": offset, "length": length}
    match_dict["sentence"] = request_string
    match_dict["type"] = {"typeName": "Hint"}
    match_dict["rule"] = {
        "id": "DEEP_VERB__" + append_id,
        "subId": 0,
        "sourceFile": "not well defined",
        "tokenizer": value["tokenizer"],
        "description": "Deep learning rules for verb, replace, comma, etc.",
        "issueType": "grammar",
        "category": {"id": "SymFree_DEEP", "name": "Deep learning rules (SymFree)"},
    }
    match_dict["ignoreForIncompleteSentence"] = False
    match_dict["contextForSureMatch"] = -1
    json_output["matches"].append(match_dict)

json_output


# %%
from gectorPredict.utils.helpers import DECODE_VERB_DICT, DECODE_VERB_DICT_MULTI

# %%
DECODE_VERB_DICT["farei_VMI1S_VMI3P"]
# %%
DECODE_VERB_DICT_MULTI["farei_VMI1S_VMI3P"]
# %%
