"""Set up Environment.
不太清楚原理是什么"""

import compaction.utils.logging as logging

_ENV_SETUP_DONE = False


def setup_environment():
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
