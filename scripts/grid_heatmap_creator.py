import sys
import os
import gym
from ast import literal_eval as make_tuple
from PIL import Image, ImageDraw, ImageFont
import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
sys.path.append('../src')
from lib_domain import *

padding = 2
cell_size = 80

black = (0, 0, 0)
red = (180, 0, 0)
light_gray = (221, 221, 221)
white = (255, 255, 255)
orange = (255, 153, 51)
green = (76, 153, 0)
purple = (153, 0, 153)


def parse_state(domain_name, state_string):
    if 'load_unload' in domain_name:
        return (np.fromstring(state_string[1:-1], dtype=int, sep=' ')[0], 0)
    elif 'tree_maze' in domain_name:
        x, y, = 0, 0
        size, offset = None, None
        if 'tree_maze-v2' == domain_name:
            size = 5
            offset = 4
            y = 7
        elif 'tree_maze-v3' == domain_name:
            size = 5
            offset = 16
            y = 31

        state_string = state_string.replace(', ', ',')
        state_string = state_string.replace(' ', ',')
        state_string = state_string.replace('[', '(')
        state_string = state_string.replace(']', ')')
        before_branch, done, _, _ = make_tuple(state_string)
        x = len(done) * (size - 1) + size - before_branch
        for d in done:
            if d == 0:
                y -= offset
            elif d == 2:
                y += offset
            offset //= 2
        return (x, y)
    return tuple(np.fromstring(state_string[1:-1], dtype=int, sep=' ')[:2])


def get_grid(domain_name):
    if domain_name == 'tree_maze-v2':
        grid_size = (15, 15)
        grid = ["&&&&&&&&&&&&&&&",
                "&&&&&&&&&_____&",
                "&&&&&&&&&|&&&&&",
                "&&&&&_____&&&&&",
                "&&&&&|&&&|&&&&&",
                "&&&&&|&&&_____&",
                "&&&&&|&&&&&&&&&",
                "&_____&&&&&&&&&",
                "&&&&&|&&&&&&&&&",
                "&&&&&|&&&_____&",
                "&&&&&|&&&|&&&&&",
                "&&&&&_____&&&&&",
                "&&&&&&&&&|&&&&&",
                "&&&&&&&&&_____&",
                "&&&&&&&&&&&&&&&"]
        return grid_size, grid

    elif domain_name == "tree_maze-v3":
        grid_size = (23, 63)
        grid = ["&&&&&&&&&&&&&&&&&&&&&&&",
                "&&&&&&&&&&&&&&&&&_____&",
                "&&&&&&&&&&&&&&&&&|&&&&&",
                "&&&&&&&&&&&&&_____&&&&&",
                "&&&&&&&&&&&&&|&&&|&&&&&",
                "&&&&&&&&&&&&&|&&&_____&",
                "&&&&&&&&&&&&&|&&&&&&&&&",
                "&&&&&&&&&_____&&&&&&&&&",
                "&&&&&&&&&|&&&|&&&&&&&&&",
                "&&&&&&&&&|&&&|&&&_____&",
                "&&&&&&&&&|&&&|&&&|&&&&&",
                "&&&&&&&&&|&&&_____&&&&&",
                "&&&&&&&&&|&&&&&&&|&&&&&",
                "&&&&&&&&&|&&&&&&&_____&",
                "&&&&&&&&&|&&&&&&&&&&&&&",
                "&&&&&_____&&&&&&&&&&&&&",
                "&&&&&|&&&|&&&&&&&&&&&&&",
                "&&&&&|&&&|&&&&&&&_____&",
                "&&&&&|&&&|&&&&&&&|&&&&&",
                "&&&&&|&&&|&&&_____&&&&&",
                "&&&&&|&&&|&&&|&&&|&&&&&",
                "&&&&&|&&&|&&&|&&&_____&",
                "&&&&&|&&&|&&&|&&&&&&&&&",
                "&&&&&|&&&_____&&&&&&&&&",
                "&&&&&|&&&&&&&|&&&&&&&&&",
                "&&&&&|&&&&&&&|&&&_____&",
                "&&&&&|&&&&&&&|&&&|&&&&&",
                "&&&&&|&&&&&&&_____&&&&&",
                "&&&&&|&&&&&&&&&&&|&&&&&",
                "&&&&&|&&&&&&&&&&&_____&",
                "&&&&&|&&&&&&&&&&&&&&&&&",
                "&_____&&&&&&&&&&&&&&&&&",
                "&&&&&|&&&&&&&&&&&&&&&&&",
                "&&&&&|&&&&&&&&&&&_____&",
                "&&&&&|&&&&&&&&&&&|&&&&&",
                "&&&&&|&&&&&&&_____&&&&&",
                "&&&&&|&&&&&&&|&&&|&&&&&",
                "&&&&&|&&&&&&&|&&&_____&",
                "&&&&&|&&&&&&&|&&&&&&&&&",
                "&&&&&|&&&_____&&&&&&&&&",
                "&&&&&|&&&|&&&|&&&&&&&&&",
                "&&&&&|&&&|&&&|&&&_____&",
                "&&&&&|&&&|&&&|&&&|&&&&&",
                "&&&&&|&&&|&&&_____&&&&&",
                "&&&&&|&&&|&&&&&&&|&&&&&",
                "&&&&&|&&&|&&&&&&&_____&",
                "&&&&&|&&&|&&&&&&&&&&&&&",
                "&&&&&_____&&&&&&&&&&&&&",
                "&&&&&&&&&|&&&&&&&&&&&&&",
                "&&&&&&&&&|&&&&&&&_____&",
                "&&&&&&&&&|&&&&&&&|&&&&&",
                "&&&&&&&&&|&&&_____&&&&&",
                "&&&&&&&&&|&&&|&&&|&&&&&",
                "&&&&&&&&&|&&&|&&&_____&",
                "&&&&&&&&&|&&&|&&&&&&&&&",
                "&&&&&&&&&_____&&&&&&&&&",
                "&&&&&&&&&&&&&|&&&&&&&&&",
                "&&&&&&&&&&&&&|&&&_____&",
                "&&&&&&&&&&&&&|&&&|&&&&&",
                "&&&&&&&&&&&&&_____&&&&&",
                "&&&&&&&&&&&&&&&&&|&&&&&",
                "&&&&&&&&&&&&&&&&&_____&",
                "&&&&&&&&&&&&&&&&&&&&&&&"]
        return grid_size, grid

    env = gym.make(domain_name)
    domain_module = importlib.import_module(env.__module__, package="src")

    return env.grid_size, env.grid


if ( len(sys.argv) == 3 ):
    domain_name = sys.argv[1]
    debug_folder = sys.argv[2]

    number_of_experiments = 0
    state_memorize_values = {}
    if os.path.isdir(debug_folder):
        for file in os.listdir(debug_folder):
            if file.endswith(".rlhist"):
                print("Reading", file, "...")
                number_of_experiments += 1
                state_memorize_counts = {}
                state_visit_counts = {}

                hfile = os.path.join(debug_folder, file)
                hmfile = hfile.replace("rlhist", "rlmhist")
                memory_lines = None
                if os.path.exists(hmfile):
                    mf = open(hmfile, 'r')
                    memory_lines = mf.readlines()

                with open(hfile, 'r') as f:
                    lines = f.readlines()
                    headers = lines[0].strip().split('\t')
                    m_headers = memory_lines[0].strip().split('\t')
                    prev_state = None
                    for line_index in range(1, len(lines)):
                        parts = lines[line_index].strip().split('\t')
                        state = None
                        ma = None

                        if 's' in headers:
                            state = parse_state(domain_name, parts[headers.index('s')])
                        if 'ma' in m_headers:
                            m_parts = memory_lines[line_index].strip().split('\t')
                            ma = m_parts[m_headers.index('ma')]

                        if state not in state_visit_counts.keys():
                            state_visit_counts[state] = 0
                            state_memorize_counts[state] = 0

                        state_visit_counts[state] += 1
                        if ma == 'P':
                            state_memorize_counts[prev_state] += 1

                        prev_state = state

                max_count = max(state_memorize_counts.values())
                for s, c in state_visit_counts.items():
                    if s not in state_memorize_values.keys():
                        state_memorize_values[s] = []
                    state_memorize_values[s].append(state_memorize_counts[s] / c)


    avg_memorize_values = {}
    for s in sorted(state_memorize_values.keys()):
        vs = state_memorize_values[s]
        avg_memorize_values[s] = sum(vs) / number_of_experiments
        print(s, avg_memorize_values[s])

    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",[(0, '#eb9872'), (1, '#532464')])

    # Make a normalizer that will map the time values from
    cnorm = mcol.Normalize(vmin=min(avg_memorize_values.values()),vmax=max(avg_memorize_values.values()))

    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    grid_size, grid = get_grid(domain_name)

    number_of_rows = grid_size[1]
    number_of_columns = grid_size[0]

    width = ((cell_size + 2*padding) * number_of_columns + padding)
    height = (cell_size + 2*padding) * number_of_rows + padding

    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, height), fill=black)

    row_pixel = padding
    for r in range(number_of_rows):
        column_pixel = padding
        for c in range(number_of_columns):
            if grid[r][c] == '|':
                #draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=light_gray, outline=black)
                draw.line([(column_pixel + cell_size//2, row_pixel), (column_pixel + cell_size//2, row_pixel + cell_size)], fill=black, width=5)
            elif grid[r][c] == '&':
                print()
            elif grid[r][c] == 'X':
                if r == 0 or r == number_of_rows-1 or c == 0 or c == number_of_columns-1:
                    draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=black, outline=black)
                else:
                    draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=light_gray, outline=black)

                    # draw obstacle
                    pxl = 0
                    while ( pxl < cell_size ):
                        if ( ((pxl+2) / 7) % 2 == 1 ):
                            draw.line([(column_pixel + pxl, row_pixel), (column_pixel + cell_size, row_pixel + cell_size - pxl)], fill=black, width=5)
                        pxl += 1

                    pxl = 0
                    while ( pxl < cell_size ):
                        if ( ((pxl+2) / 7) % 2 == 0 ):
                            draw.line([(column_pixel, row_pixel + pxl), (column_pixel + cell_size - pxl, row_pixel + cell_size)], fill=black, width=5)
                        pxl += 1
            else:
                value = avg_memorize_values[(c, r)] if (c, r) in avg_memorize_values.keys() else 0
                clr = cpick.to_rgba(value)
                color_value = (int(clr[0] * 255), int(clr[1] * 255), int(clr[2] * 255))
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=color_value, outline=black)

            column_pixel += 2*padding + cell_size
        row_pixel += 2*padding + cell_size

    heatmap_name = debug_folder.split('/')[-2]
    img.save(os.path.join(debug_folder, heatmap_name + "_memory_heatmap") + ".png")

else:
	print("Wrong number of arguments!")
	print("Usage: python grid_heatmap_creator.py <GRID DOMAIN NAME> <DEBUG FOLDER>")
