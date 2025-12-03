from common.logging import log


DEBUG = False


def tcpu(t):
    return t.detach().cpu().item()


def tinfo(tag, t, verbose=DEBUG):
    # torch.tensors
    if verbose:
        log('tinfo:', tag, t.shape, tcpu(t.min()), tcpu(t.mean()), tcpu(t.max()))


def ainfo(tag, t):
    log('ainfo:', tag, t.shape, t.min(), t.mean(), t.max())
