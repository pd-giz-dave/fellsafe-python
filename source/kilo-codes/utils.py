""" Random useful stuff """

import os
import pathlib
import canvas

class Logger:
    """ crude logging system that saves to a file and prints to the console """

    def __init__(self, log_file: str, folder: str='.', context: str=None, prefix='  '):
        # make sure the destination folder exists
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        self.log_file = '{}/{}'.format(folder, log_file)
        self.log_handle = None
        if context is None:
            pathname, _ = os.path.splitext(log_file)
            _, context = os.path.split(pathname)
        self.context: [(str, str)] = [(context, folder)]
        self.prefix = prefix  # when logging multi-line messages prefix all lines except the first with this
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
        if msg == '\n':
            # caller just wants a blank line
            lines = ['']
        else:
            lines = msg.split('\n')
        for line, text in enumerate(lines):
            if line > 0:
                prefix = self.prefix
            else:
                prefix = ''
            log_msg = '{}: {}{}'.format(self.context[0][0], prefix, text)
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

class Stats:
    """ log quantized value statistics """

    def __init__(self, buckets: int, value_range=(0,1), number_format='{:.2f}'):
        self.buckets       = buckets                          # how many buckets to quantize values into
        self.number_format = number_format                    # how to show numbers in logs
        self.value_min     = value_range[0]                   # expected minimum value
        self.value_max     = value_range[1]                   # expected maximum value
        self.value_span    = self.value_max - self.value_min  # nuff said?
        self.value_delta   = self.value_span / self.buckets   # value range in each bucket
        self.counts        = [0 for _ in range(buckets+1)]    # count of values in each bucket
        if self.value_span <= 0:
            raise Exception('value min must be >= 0 and value max must be > min (given {})'.format(value_range))

    def reset(self):
        """ clear stats to 0 """
        for bucket in range(len(self.counts)):
            self.counts[bucket] = 0

    def normalise(self, value):
        """ normalize the given value into the range 0..1 """
        norm  = min(max(value, self.value_min), self.value_max)  # ?..?     --> min..max
        norm -= self.value_min                                   # min..max --> 0..span
        norm /= self.value_span                                  # 0..span  --> 0..1
        return norm

    def bucket(self, value):
        """ return the bucket number for the given value """
        if value is None:
            return None
        norm   = self.normalise(value)
        bucket = int(norm * self.buckets)
        if bucket < 0 or bucket >= len(self.counts):
            raise Exception('bucket not in range 0..{} for value {}'.format(len(self.counts)-1, value))
        return bucket

    def span(self, bucket):
        """ return the value span (from(>=)-->to(<)) represented by the given bucket """
        span = (self.value_min + (bucket * self.value_delta), self.value_min + ((bucket+1) * self.value_delta))
        return span

    def count(self, value):
        bucket = self.bucket(value)
        if bucket is None:
            return
        self.counts[bucket] += 1

    def show(self, separator=', '):
        """ return a string showing the current stats """
        msg = ''
        for bucket, count in enumerate(self.counts):
            if count == 0:
                continue
            bucket_min, bucket_max = self.span(bucket)
            value_min = self.number_format.format(bucket_min)
            value_max = self.number_format.format(bucket_max)
            msg = '{}{}{}..{}-{}'.format(msg, separator, value_min, value_max, count)
        return msg[len(separator):]

class Frequencies:
    """ count the frequency of something within a set """

    def __init__(self, number_scale=100, number_format='{:.2f}%'):
        self.number_scale  = number_scale
        self.number_format = number_format
        self.reset()

    def reset(self):
        self.set   = {}
        self.size  = 0
        self.total = 0

    def count(self, item):
        if self.set.get(item) is None:
            # not seen this before
            self.set[item] = 1
            self.size += 1
        else:
            # seen before
            self.set[item] += 1
        self.total += 1

    def show(self, separator='\n'):
        """ return a string representing the counts """
        msg = ''
        for item, count in self.set.items():
            ratio = count / self.total
            freq = self.number_format.format(ratio * self.number_scale)
            msg = '{}{}{}: {} ({})'.format(msg, separator, item, count, freq)
        return msg[len(separator):]

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
