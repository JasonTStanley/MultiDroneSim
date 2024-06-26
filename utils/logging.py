"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/logging.py
"""

import logging
import sys


try:
    import coloredlogs

    coloredlogs.install()
except BaseException:
    pass

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)