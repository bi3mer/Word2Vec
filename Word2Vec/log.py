from colorama import Fore, Style

def info(text, verbose=True):
    if verbose:
        print(f"{Fore.LIGHTBLUE_EX}INFO:{Style.RESET_ALL} {text}")

def warning(text, verbose=True):
    if verbose:
        print(f"{Fore.YELLOW}WARNING:{Style.RESET_ALL} {text}")

def error(text, verbose=True):
    if verbose:
        print(f"{Fore.RED}ERROR:{Style.RESET_ALL} {text}")

def new_line(num_times=1):
    for _ in range(num_times):
        print()