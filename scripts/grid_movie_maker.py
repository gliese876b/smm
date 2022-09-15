import sys
import os
import gym
from ast import literal_eval as make_tuple
from PIL import Image, ImageDraw, ImageFont
import importlib
import numpy as np
import moviepy.editor as mpy
sys.path.append('../src')
from lib_domain import *

padding = 2
cell_size = 80
queue_length = 1

black = (0, 0, 0)
red = (180, 0, 0)
light_gray = (221, 221, 221)
white = (255, 255, 255)
orange = (255, 153, 51)
green = (76, 153, 0)
purple = (153, 0, 153)
frames = []

def draw_observation(draw, observation, x, y, is_goal_obs, border):
    size = (cell_size + 2*padding) * 3 + padding

    if border:
        v = int(255 * border)
        w = 5
        draw.rectangle((x - w, y - w, x + size + w, y + size + w), outline=(v, v, v), width=w)

    mapping = {(1, 0): 0, (0, 1): 3, (2, 1): 1, (1, 2): 2}
    row_pixel = y + padding
    for c in range(3):
        column_pixel = x + padding
        for r in range(3):
            if (r, c) == (1, 1):
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=green if is_goal_obs else white, outline=black)
            elif (r, c) in mapping.keys():
                o_f = mapping[(r, c)]
                if observation[o_f] == 1:   # the cell is full
                    draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=light_gray, outline=black)
                    
                    # draw obstacle
                    pxl = 0
                    while ( pxl < cell_size  ):
                        if ( ((pxl+2) / 7) % 2 == 1 ):
                            draw.line([(column_pixel + pxl, row_pixel), (column_pixel + cell_size, row_pixel + cell_size - pxl)], fill=black, width=5)
                        pxl += 1

                    pxl = 0
                    while ( pxl < cell_size ):
                        if ( ((pxl+2) / 7) % 2 == 0 ):
                            draw.line([(column_pixel, row_pixel + pxl), (column_pixel + cell_size - pxl, row_pixel + cell_size)], fill=black, width=5)
                        pxl += 1
                    
                else:
                    draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=white, outline=black)
                
            column_pixel += 2*padding + cell_size
        row_pixel += 2*padding + cell_size
    

if ( len(sys.argv) == 3 ):
    domain_name = sys.argv[1]
    history_file = sys.argv[2]
    memory_history_file = history_file.replace("rlhist", "rlmhist")

    env = gym.make(domain_name)
    domain_module = importlib.import_module(env.__module__, package="src")
    is_partially_observable = "_get_one_agent_observation" in dir(env)
    # FIXME
    is_partially_observable = True

    number_of_rows = env.grid_size[1]
    number_of_columns = env.grid_size[0]

    po_size = (cell_size + 2*padding) * 6 + 2 * cell_size + 10*padding
    width = ((cell_size + 2*padding) * number_of_columns + padding) + (po_size if is_partially_observable else 0)
    height = (cell_size + 2*padding) * number_of_rows + padding

    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, height), fill=black)
    
    row_pixel = padding
    for r in range(number_of_rows):
        column_pixel = padding
        for c in range(number_of_columns):
            if env.grid[r][c] != "X":
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=white, outline=black)
            else:
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

            column_pixel += 2*padding + cell_size
        row_pixel += 2*padding + cell_size
    
    '''
    # fonts for cell_size = 40
    fnt = ImageFont.truetype("./fonts/PlayfairDisplay-Regular.ttf", 40)
    fnt_bold = ImageFont.truetype("./fonts/PlayfairDisplay-Bold.ttf", 40)
    fnt_italic = ImageFont.truetype("./fonts/PlayfairDisplay-Italic.ttf", 40)
    fnt_italic_small = ImageFont.truetype("./fonts/PlayfairDisplay-Italic.ttf", 24)
    '''
    
    fnt = ImageFont.truetype("./fonts/PlayfairDisplay-Regular.ttf", 60)
    fnt_bold = ImageFont.truetype("./fonts/PlayfairDisplay-Bold.ttf", 60)
    fnt_italic = ImageFont.truetype("./fonts/PlayfairDisplay-Italic.ttf", 60)
    fnt_italic_small = ImageFont.truetype("./fonts/PlayfairDisplay-Italic.ttf", 42)
    
    for g in env.li_goal_states:
        g_x, g_y = g
        draw.multiline_text((g_x * (cell_size + 2*padding) + 10*padding, g_y * (cell_size + padding) + 4*padding), 'G', font=fnt_bold, fill=green)
    
    if is_partially_observable:
        # memory column
        memory_column_x = (cell_size + 2*padding) * number_of_columns + 4 * padding
        memory_column_y = (cell_size + 2*padding) * 3 + int(cell_size / 2.0)
        draw.multiline_text((memory_column_x - 22*padding, memory_column_y + int(1.5 * cell_size) - 17*padding), '<', font=fnt, fill=white)
        draw.multiline_text((memory_column_x + 3*cell_size + 10*padding, memory_column_y + int(1.5 * cell_size) - 17*padding), '>', font=fnt, fill=white)
        draw.multiline_text((memory_column_x + cell_size + 10 * padding, 95 * padding), 'm', font=fnt_italic, fill=white)
        draw.multiline_text((memory_column_x + cell_size + 34 * padding, 110 * padding), 't', font=fnt_italic_small, fill=white)

        # observation column
        observation_column_x = memory_column_x + 3 * cell_size + 60 * padding 
        observation_column_y = memory_column_y
        draw.multiline_text((observation_column_x + cell_size + 15 * padding, 95 * padding), "o", font=fnt_italic, fill=white)
        draw.multiline_text((observation_column_x + cell_size + 30 * padding, 110 * padding), 't', font=fnt_italic_small, fill=white)
    
    memory_lines = None
    if os.path.exists(memory_history_file):
        mf = open(memory_history_file, 'r')
        memory_lines = mf.readlines()
    
    history = []
    queue = []
    frames += [np.asarray(img)]	
    with open(history_file, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split('\t')
        m_headers = memory_lines[0].strip().split('\t')
        for line_index in range(1, len(lines)):
            parts = lines[line_index].strip().split('\t')
            state = None
            observation = None
            memory = None
            ma = None
            if is_partially_observable and 'o' in headers:
                observation = np.fromstring(parts[headers.index('o')][1:-1], dtype=np.int, sep=' ')
            if 's' in headers:
                state = np.fromstring(parts[headers.index('s')][1:-1], dtype=np.int, sep=' ')
            if 'x' in m_headers:
                m_parts = memory_lines[line_index].strip().split('\t')
                m_tuple = make_tuple(m_parts[m_headers.index('x')])
                
                if len(m_tuple) > 1 and type(m_tuple[-1]) is tuple:
                    m = m_tuple[:-1]
                    memory = np.asarray(m[-1][0], dtype=np.int)      
            if 'ma' in m_headers:
                m_parts = memory_lines[line_index].strip().split('\t')
                ma = m_parts[m_headers.index('ma')]
                
            history.append((state, observation, memory, ma))
    		
    prev_obs_in_memory = None
    border_value = 1.0
    for h in range(len(history)):			 
        time_area = (memory_column_x + 3 * cell_size, cell_size)
        draw.rectangle((time_area[0], time_area[1], time_area[0] + 5*cell_size, time_area[1] + 1*cell_size), fill=black, outline=black)
        draw.multiline_text((time_area[0] + 10 * padding, time_area[1] + 2 * padding), 't = {}'.format(h), font=fnt, fill=white)

        if h > 0:
            state, observation, memory, ma = history[h-1]
            states_in_memory = [tuple(x[0]) for x in queue]

            if state is not None and tuple(state) not in states_in_memory:
                column_pixel = state[0] * (2*padding + cell_size) + padding
                row_pixel = state[1] * (2*padding + cell_size) + padding
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=white, outline=black)

        if ( len(queue) > queue_length ):
            state, observation, memory, ma = queue[0]
            if state is not None:
                column_pixel = state[0] * (2*padding + cell_size) + padding
                row_pixel = state[1] * (2*padding + cell_size) + padding
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=white, outline=black)
                del queue[0]
		
        for cell in range(len(queue)):
            state, obs_in_memory, memory, ma = queue[cell]

            if state is not None:
                color_value = int(160 * (1.0 - float(len(queue) - cell - 1) / float(queue_length)))	
                column_pixel = state[0] * (2*padding + cell_size) + padding
                row_pixel = state[1] * (2*padding + cell_size) + padding
                shrink_size = 7 * padding
                draw.rectangle((column_pixel, row_pixel, column_pixel + cell_size, row_pixel + cell_size), fill=orange, outline=black)                
                
                if cell == len(queue) - 1:
                    if is_partially_observable and obs_in_memory is not None and len(obs_in_memory) == 4:
                        if np.array_equal(obs_in_memory, prev_obs_in_memory):
                            border_value *= 0.5
                        else:
                            border_value = 1.0
                        prev_obs_in_memory = obs_in_memory
                        draw_observation(draw, obs_in_memory, memory_column_x, memory_column_y, False, border_value) 

        state, observation, memory, ma = history[h]
        if state is not None:
            color_value = int(160)	
            column_pixel = state[0] * (2*padding + cell_size) + padding
            row_pixel = state[1] * (2*padding + cell_size) + padding
            shrink_size = 7 * padding
            draw.ellipse((column_pixel + shrink_size, row_pixel + shrink_size, column_pixel + cell_size - shrink_size, row_pixel + cell_size - shrink_size), fill=(color_value, 0, color_value), outline=black)

            if is_partially_observable and observation is not None and len(observation) == 4:
                draw_observation(draw, observation, observation_column_x, observation_column_y, tuple(state) in env.li_goal_states, None)
        

        frames.append( np.asarray(img) )
        if h == 0 or h == len(history) - 1:
            for _ in range(10):
                frames.append( np.asarray(img) )
        
        state, observation, memory, ma = history[h]
        if state is not None and ma == 'P':
            queue.append(history[h])
    
    video_fps = 2
    clip = mpy.ImageSequenceClip(frames, fps=video_fps)
    clip.write_videofile(history_file + ".avi", fps=video_fps, codec="png")
    
	
else:
	print("Wrong number of arguments!")
	print("Usage: python grid_movie_maker.py <GRID DOMAIN NAME> <HISTORY>.rlhist")
