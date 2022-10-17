import warnings

import pytorch_tao as tao


def tell(score: float):
    if tao.study is None:
        warnings.warn("`tao.tell` is skipped when tao is not in tune mode")
        return
    tao.study.tell(tao.trial, score)
