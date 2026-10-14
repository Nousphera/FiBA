# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import os
import logging
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import pickle


def init_logger(args, log_lvl=logging.INFO):
	# Create project dir.
	project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	# Create log dir if not exists.
	os.makedirs(f"{os.path.join(project_dir,args.log_dir)}", exist_ok=True)
	# Derive logger name from arguments
	logger_name = "{}/{}_{}_{}_{}_{}_{}.log".format(os.path.join(project_dir,args.log_dir), os.path.basename(project_dir), \
		args.dataset, args.data_split, int(100*args.skewness), int(100*args.participation), args.num_rounds)
	print(f"\nLogging progress to {logger_name}\n")
	logger = logging.getLogger('flwr')
	logger.setLevel(log_lvl)
	file_log = logging.FileHandler(logger_name, mode='w')
	file_log.setFormatter(logging.Formatter("flwr | %(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"))
	logger.addHandler(file_log)
	return logger


def select_fraction_of_numbers(n, fraction, sd):
    """Select a random fraction of numbers (to be seleted as attackers).

    Keyword arguments:
    n -- The total amount of participants
    fraction -- the fraction of participants to be selected as attackers
    """
    if fraction < 0 or fraction > 1:
        raise ValueError("Fraction should be between 0 and 1")

    x = int(n * fraction)
    random.seed(sd)
    return random.sample(range(n), x)

def split_image(w, h, num_parts):
    # Calculate the number of rows and columns
    num_cols = int(num_parts**0.5)
    num_rows = (num_parts + num_cols - 1) // num_cols

    # Calculate the size of each rectangle
    rectangle_width = w // num_cols
    rectangle_height = h // num_rows

    # Adjust the width of each rectangle to account for any remainder
    rectangle_widths = [rectangle_width + (1 if i < w % num_cols else 0) for i in range(num_cols)]

    rectangles = []

    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
            x1 = sum(rectangle_widths[:col])
            y1 = row * rectangle_height
            x2 = x1 + rectangle_widths[col]
            y2 = y1 + rectangle_height

			 # Skip creating extra rectangles beyond the specified num_parts
            if len(rectangles) < num_parts:
                rectangles.append((x1, y1, x2, y2))
            else:
                # Append any remaining space to the last rectangle
                rectangles[-1] = (rectangles[-1][0], rectangles[-1][1], w, h)
                break


    return rectangles
