""" Random useful stuff """

class Logger:
    """ crude logging system that saves to a file and prints to the console """

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.log_handle = None
        self.context: [str] = [self.log_file]
        self.log('open')

    def __del__(self):
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def log(self, msg: str=None):
        if msg is None:
            self.log_handle.close()
            self.log_handle = None
            return
        if self.log_handle is None:
            self.log_handle = open(self.log_file, 'w')
        lines = msg.split('\n')
        for line in lines:
            log_msg = '{}: {}'.format(self.context[0], line)
            self.log_handle.write('{}\n'.format(log_msg))
            self.log_handle.flush()
            print(log_msg)

    def push(self, context):
        self.context.insert(0, context)

    def pop(self):
        self.context.pop(0)
