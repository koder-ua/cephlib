import sys

if sys.version_info > (3, 0):
    from .p3 import *
    import queue
    import logging.config as logging_config
else:
    from .p2 import *
    import Queue as queue
    from logging import config as logging_config
