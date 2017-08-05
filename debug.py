#!/usr/bin/env python3

import sys
import IPython.core.ultratb

sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)
