import os

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Color:
    # ANSI escape sequences for text colors
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def printw(message):
    print(f"{Color.YELLOW}Warning: {message}{Color.END}")

def printe(message):
    print(f"{Color.RED}Error: {message}{Color.END}")

def printi(message):
    print(f"{Color.BLUE}Info: {message}{Color.END}")

def getenv(key: str) -> bool:
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v is not None and v.lower() == '1':
            return True
    return False