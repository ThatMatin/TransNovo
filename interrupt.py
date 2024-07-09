import signal
from logger import setup_logger

logger = setup_logger(__name__)

class InterruptHandler:
    def __init__(self, *funcs):
        self.__interrupted = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        self.funcs = funcs

    def handle_signal(self, signum, frame):
        logger.debug("request to interrupt, wait for gracious shutdown")
        self.__interrupted = True
        for f in self.funcs:
            f()

    def is_interrupted(self) -> bool:
        return self.__interrupted
