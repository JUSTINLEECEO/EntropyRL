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
    cider_scores = [coco_eval.imgToEval[i]["CIDEr"] for i in range(len(reward_inputs))]
    
    return [{"overall": score, "CIDEr": score} for score in cider_scores]
