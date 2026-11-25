"""Use the upstream option extractor and give unrecognized answers zero credit."""

from lmms_eval.tasks.hrbench.utils import hrbench_doc_to_options, hrbench_evaluator


def process_results(doc, results):
    prediction = hrbench_evaluator.can_infer(results[0], hrbench_doc_to_options(doc))
    record = {
        "index": doc["index"],
        "cycle_category": doc["cycle_category"],
        "gpt_score": int(prediction == doc["answer"].strip().upper()),
    }
    return {doc["category"]: record, "average": record}
