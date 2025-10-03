from typing import Any, Dict, List
from COCOeval.eval import COCOEvalCap

def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    # mini COCO object
    class DummyCOCO:
        def __init__(self, imgToAnns):
            self.imgToAnns = imgToAnns
        def getImgIds(self):
            return list(self.imgToAnns.keys())
    gts = {}
    res = {}
    for i, reward_input in enumerate(reward_inputs):
        gts[i] = [{"caption": cap} for cap in reward_input["ground_truth"]]
        res[i] = [{"caption": reward_input["response"]}]
    coco = DummyCOCO(gts)
    cocoRes = DummyCOCO(res)
    coco_eval = COCOEvalCap(coco, cocoRes)
    coco_eval.evaluate()

    bleu_1_scores = [coco_eval.imgToEval[i]["Bleu_1"] for i in range(len(reward_inputs))]
    bleu_2_scores = [coco_eval.imgToEval[i]["Bleu_2"] for i in range(len(reward_inputs))]
    bleu_3_scores = [coco_eval.imgToEval[i]["Bleu_3"] for i in range(len(reward_inputs))]  
    bleu_4_scores = [coco_eval.imgToEval[i]["Bleu_4"] for i in range(len(reward_inputs))]
    cider_scores = [coco_eval.imgToEval[i]["CIDEr"] for i in range(len(reward_inputs))]
    meteor_scores = [coco_eval.imgToEval[i]["METEOR"] for i in range(len(reward_inputs))]
    rouge_scores = [coco_eval.imgToEval[i]["ROUGE_L"] for i in range(len(reward_inputs))]
    
    payload = []
    for i in range(len(cider_scores)):
        payload.append({
            "overall": cider_scores[i],
            "CIDEr": cider_scores[i],
            "BLEU_1": bleu_1_scores[i],
            "BLEU_2": bleu_2_scores[i],
            "BLEU_3": bleu_3_scores[i],
            "BLEU_4": bleu_4_scores[i],
            "METEOR": meteor_scores[i],
            "ROUGE": rouge_scores[i]
        })
    return payload
    # return [{"overall": score, "CIDEr": score} for score in cider_scores]
