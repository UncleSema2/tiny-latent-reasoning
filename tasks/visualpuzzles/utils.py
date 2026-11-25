"""Full VisualPuzzles split with direct multiple-choice answers."""

from lmms_eval.tasks.unig2u.visualpuzzles import utils

images = utils.VisualPuzzles_doc_to_visual
prompt = utils.VisualPuzzles_doc_to_text


def score(doc, results):
    response = results[0].strip()
    choices = "ABCD"
    if response.upper() in tuple(choices):
        prediction = response.upper()
    else:
        options = dict(zip(choices, doc["options"])) if doc["options"] else None
        prediction = utils.parse_response(response, choices, options)
    return {"accuracy": float((prediction or "").upper() == doc["answer"])}
