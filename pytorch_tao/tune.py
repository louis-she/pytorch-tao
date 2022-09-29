import pytorch_tao as tao


def tell(score: float):
    if tao.study is None:
        raise ValueError()
    tao.study.tell(tao.trial, score)
