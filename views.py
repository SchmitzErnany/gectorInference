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
    replacements_to_json,
)
from gectorPredict.gector.gec_model import GecBERTModel

args = {
    "vocab_path": "gectorPredict/MODEL_DIR/vocabulary/",
    "model_path": ["gectorPredict/MODEL_DIR/best.th"],
    "max_len": 50,
    "min_len": 2,
    "iteration_count": 5,
    "min_error_probability": {
        "all": 0.8,
        "comma": 0.8,
        "addcrase": 0.6,
        "uppercase_into_3S": 0.97,
    },
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

request_string = "está cadeira esta aqui."
repl = predict_for_paragraph(
    request_string,
    model,
    tokenizer_method="split+spacy",
)
json_output = replacements_to_json(
    version="1.2",
    request_string=request_string,
    replacements_dictionary=repl,
)
json_output

# %%
