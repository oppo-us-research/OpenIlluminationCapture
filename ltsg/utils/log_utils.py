#  created by Isabella Liu (lal005@ucsd.edu) at 2023/03/20 19:50.

from loguru import logger


def log_args(args):
    logger.info('*' * 10 + "args" + "*" * 10)
    for k in vars(args):
        logger.info(f"{k}: {getattr(args, k)}")
    logger.info('*' * 20)