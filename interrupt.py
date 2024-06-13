import signal

class InterruptHandler:
    def __init__(self):
        self.__interrupted = False
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        print("Okay, Okay, let me clean up after this training loop ends...")
        self.__interrupted = True

    def is_interrupted(self):
        return self.__interrupted
