""" Random useful stuff """

import os
import pathlib
import canvas

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

    def depth(self):
        return len(self.context)

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
        canvas.unload(image, filename)

        self.log('{}: image saved as: {}'.format(file, filename))


def _unload(image, source=None, file='', target=(0,0), logger=None):
    """ unload the given image with a name that indicates its source and context,
        file is the file base name to save the image as,
        target identifies the x,y of the primary locator the image represents,
        target of 0,0 means no x/y identification for the image name,
        """

    folder = image_folder(source=source, target=target)
    logger.draw(image, folder=folder, file=file)

def image_folder(source=None, target=(0,0)):
    """ build folder name for diagnostic images for the given target """
    if target[0] > 0 and target[1] > 0:
        # use a sub-folder for this image
        folder = '{:.0f}x{:.0f}y'.format(target[0], target[1])
    else:
        folder = ''
    if source is not None:
        # construct parent folder to save images in for this source
        pathname, _ = os.path.splitext(source)
        _, basename = os.path.split(pathname)
        folder = '{}{}'.format(basename, folder)
    return folder
