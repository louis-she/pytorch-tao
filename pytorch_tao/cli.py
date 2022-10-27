from pytorch_tao import core


def main():
    args = core.parse_tao_args()
    core.dispatch(args)
