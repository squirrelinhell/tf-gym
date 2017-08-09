
import os
import sys

def print_err(_1, err, _2):
    info = "[Note: set the environment variable DEBUG to see details]"
    err = str(err)
    if len(err) >= 1:
        sys.stderr.write("Error: " + str(err) + "\n" + info + "\n")
        sys.stderr.flush()

if "DEBUG" in os.environ and os.environ["DEBUG"] not in ["", "no"]:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)
else:
    sys.excepthook = print_err
