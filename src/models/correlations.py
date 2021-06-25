import pandas as pd

def igf_compare():
    data_dir = "../../data/interim/igf/"
    failai = list(pathlib.Path(data_dir).glob("*igf.csv"))
    failai
    saveDir = pathlib.Path("../../data/interim/correlations")
    saveDir.mkdir(exist_ok=True)
    df = [pd.read_csv(x) for x in failai]
    for 

if __name__ == "__main__":
    main()
