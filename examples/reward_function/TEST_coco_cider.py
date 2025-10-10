from typing import Any, Dict, List
from COCOeval.eval import COCOEvalCap
import requests

def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    # print(reward_inputs)
    # mini COCO object
    class DummyCOCO:
        def __init__(self, imgToAnns):
            self.imgToAnns = imgToAnns
        def getImgIds(self):
            return list(self.imgToAnns.keys())
    gts = {}
    res = {}
    image_ids = {}
    for i, reward_input in enumerate(reward_inputs):
        gts[i] = [{"caption": cap} for cap in reward_input["ground_truth"]]
        res[i] = [{"caption": reward_input["response"]}]
    coco = DummyCOCO(gts)
    cocoRes = DummyCOCO(res)
    coco_eval = COCOEvalCap(coco, cocoRes)
    coco_eval.evaluate()

    # obtain COCO metrics
    bleu_1_scores = [coco_eval.imgToEval[i]["Bleu_1"] for i in range(len(reward_inputs))]
    bleu_2_scores = [coco_eval.imgToEval[i]["Bleu_2"] for i in range(len(reward_inputs))]
    bleu_3_scores = [coco_eval.imgToEval[i]["Bleu_3"] for i in range(len(reward_inputs))]  
    bleu_4_scores = [coco_eval.imgToEval[i]["Bleu_4"] for i in range(len(reward_inputs))]
    cider_scores = [coco_eval.imgToEval[i]["CIDEr"] for i in range(len(reward_inputs))]
    meteor_scores = [coco_eval.imgToEval[i]["METEOR"] for i in range(len(reward_inputs))]
    rouge_scores = [coco_eval.imgToEval[i]["ROUGE_L"] for i in range(len(reward_inputs))]

    overall_chair_i = None
    overall_chair_s = None

    # prepare items only if image_id exists in inputs
    if all(("image_id" in ri) for ri in reward_inputs):
        items = [{"caption": ri["response"], "image_id": int(ri["image_id"])} for ri in reward_inputs]
        server = "http://127.0.0.1:5000"
        timeout = 30
        resp = requests.post(server.rstrip('/') + '/computeCaption', json=items, headers={"Content-Type": "application/json"}, timeout=timeout)
        resp.raise_for_status()
        res_chair = resp.json()
        overall_chair_i = res_chair["overall_metrics"]["CHAIRi"]
        overall_chair_s = res_chair["overall_metrics"]["CHAIRs"]
    else:
        raise ValueError("All reward_inputs must contain 'image_id' to compute CHAIR metrics.")

    payload = []
    for i in range(len(cider_scores)):
        item = {
            "overall": cider_scores[i],
            "CIDEr": cider_scores[i],
            "BLEU_1": bleu_1_scores[i],
            "BLEU_2": bleu_2_scores[i],
            "BLEU_3": bleu_3_scores[i],
            "BLEU_4": bleu_4_scores[i],
            "METEOR": meteor_scores[i],
            "ROUGE": rouge_scores[i],
            "CHAIRi": overall_chair_i,
            "CHAIRs": overall_chair_s
        }
        payload.append(item)
    return payload
    # return [{"overall": score, "CIDEr": score} for score in cider_scores]
