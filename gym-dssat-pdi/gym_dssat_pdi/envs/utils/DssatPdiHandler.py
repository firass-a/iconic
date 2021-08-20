"""
from https://stackoverflow.com/questions/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
"""
import os
import signal


class DssatPdiHandler:
    def __enter__(self):
        os.setpgrp()

    def __exit__(self, type, value, traceback):
        try:
            os.killpg(0, signal.SIGTERM)
        except KeyboardInterrupt:
            pass