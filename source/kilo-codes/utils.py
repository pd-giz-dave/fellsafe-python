""" Random useful stuff """

import os
import pathlib
import cv2

class Logger:
    """ crude logging system that saves to a file and prints to the console """

    def __init__(self, log_file: str, folder: str='.', context: str=None):
        # make sure the destination folder exists
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        self.log_file = '{}/{}'.format(folder, log_file)
        self.log_handle = None
        if context is None:
            pathname, _ = os.path.splitext(log_file)
            _, context = os.path.split(pathname)
        self.context: [(str, str)] = [(context, folder)]
        self.log('open {}'.format(self.log_file))
        self.count = 0  # incremented for every anonymous draw call and used as a file name suffix

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
            log_msg = '{}: {}'.format(self.context[0][0], line)
            self.log_handle.write('{}\n'.format(log_msg))
            self.log_handle.flush()
            print(log_msg)

    def push(self, context=None, folder=None):
        parent_context = self.context[0][0]
        parent_folder  = self.context[0][1]
        if context is not None:
            parent_context = '{}/{}'.format(parent_context, context)
        if folder is not None:
            parent_folder = '{}/{}'.format(parent_folder, folder)
        self.context.insert(0, (parent_context, parent_folder))

    def pop(self):
        self.context.pop(0)

    def draw(self, image, folder='', file=''):
        """ unload the given image into the given folder and file,
            folder, iff given, is a sub-folder to save it in (its created as required),
            the parent folder is that given when the logger was created,
            all images are saved as a sub-folder of the parent,
            file is the file name to use, blank==invent one,
            """

        if file == '':
            file = 'draw-{}'.format(self.count)
            self.count += 1

        if folder == '':
            folder = self.context[0][1]

        # make sure the destination folder exists
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        filename = '{}/{}.png'.format(folder, file)

        # save the image
        cv2.imwrite(filename, image)

        self.log('{}: image saved as: {}'.format(file, filename))
