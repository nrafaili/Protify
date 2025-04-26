import os
import torch
import shutil
import pyfiglet
from functools import partial


torch_load = partial(torch.load, map_location='cpu', weights_only=True)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_message(message: str):
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 50
    print('\n' + '-' * terminal_width)
    print(f'\n{message}\n')
    print('-' * terminal_width + '\n')


def print_title(title: str):
    print(pyfiglet.figlet_format(title, font='3d-ascii'))


def print_done():
    print(pyfiglet.figlet_format('== Done ==', font='js_stick_letters'))


if __name__ == '__main__':
    ### Test clearing, gathers all code in the repo
    import time
    from pathlib import Path
    
    python_files = []
    for path in Path('.').rglob('*.py'):
        python_files.append(str(path))
    
    # Read and print contents of each file
    for file in python_files:
        print(f"\n=== Contents of {file} ===\n")
        with open(file, 'r') as f:
            print(f.read())
        time.sleep(0.2)
        clear_screen()
