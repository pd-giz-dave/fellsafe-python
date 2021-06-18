""" Hook the exception handler to invoke the post-mortem debugger
    See https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
    To use it just
        import debug
    at start of the (outer) python file
    """

import sys


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


sys.excepthook = info
