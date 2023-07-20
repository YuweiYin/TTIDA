# -*- coding:utf-8 -*-

__author__ = 'tylin'

from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.bleu.bleu import Bleu
from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.meteor.meteor import Meteor
from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.rouge.rouge import Rouge
from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.cider.cider import Cider
from mplug.language_evaluation.coco_caption_py3.pycocoevalcap.spice.spice import Spice

_COCO_TYPE_TO_METRIC = {
    "BLEU": (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    "METEOR": (Meteor(), "METEOR"),
    "ROUGE_L": (Rouge(), "ROUGE_L"),
    "CIDEr": (Cider(), "CIDEr"),
    "SPICE": (Spice(), "SPICE"),
}


class COCOEvalCap:
    def __init__(self, coco, cocoRes, cocoTypes, tokenization_fn=None, verbose=True):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}
        self.cocoTypes = cocoTypes
        self.tokenization_fn = tokenization_fn
        self.verbose = verbose

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer(self.tokenization_fn, verbose=self.verbose)
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [_COCO_TYPE_TO_METRIC[coco_type] for coco_type in self.cocoTypes]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing {} score...'.format(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("{}: {:3}".format(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("{}: {:3}".format(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
