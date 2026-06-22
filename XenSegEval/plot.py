import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_results(
    eval_path
):
    df = pd.read_csv(eval_path)
    if 'Method' in df.columns:
        print('WIP')