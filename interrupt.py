import signal
from logger import setup_logger

logger = setup_logger(__name__)

class InterruptHandler:
    def __init__(self):
        self.__interrupted = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        logger.debug("request to interrupt")
        self.__interrupted = True

    def is_interrupted(self) -> bool:
        return self.__interrupted
