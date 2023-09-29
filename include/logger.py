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