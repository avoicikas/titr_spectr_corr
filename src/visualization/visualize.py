import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

def plot_psds():
    data_dir = "../../data/interim/powers"
    failai = list(pathlib.Path(data_dir).glob("*.csv"))
    saveDir = pathlib.Path("../../reports/figures/powers")
    saveDir.mkdir(exist_ok=True)
    for ifile in failai:
        df = pd.read_csv(ifile)
        df.groupby(by=['subjname','ichan']).mean()
        df.subjname.unique()


if __name__ == "__main__":
    plot_psds()
