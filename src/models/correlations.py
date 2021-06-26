import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

def igf_compare_measures():
    data_dir = "../../data/interim/igf/"
    failai = list(pathlib.Path(data_dir).glob("*igf.csv"))
    failai
    df = [pd.read_csv(x) for x in failai]
    corelations =[]
    for idf in df:
        for iidf in df:
            corelations.append(idf.corrwith(iidf, axis=1).mean())
    corelations=np.reshape(corelations,[64//len(failai), len(failai)])
    sns.heatmap(corelations,cmap='coolwarm', annot=True)
    plt.xlabel([x.stem for x in failai])

def igf_meta_corr():
    data_dir = "../../data/interim/igf/"
    saveDir = pathlib.Path("../../data/interim/correlations")
    saveDir.mkdir(exist_ok=True)
    ftotalpower_ratio_igf = pd.read_csv(list(pathlib.Path(data_dir).glob("total_power_ratioigf.csv"))[0])
    ftotalpower_ratio_igf_cmean = pd.read_csv(list(pathlib.Path(data_dir).glob("total_power_ratioigf_cmean.csv"))[0])
    ftotalpower_ratio_igf_value = pd.read_csv(list(pathlib.Path(data_dir).glob("total_power_ratioigf_value.csv"))[0])
    ftotalpower_ratio_igf_value_cmean = pd.read_csv(list(pathlib.Path(data_dir).glob("total_power_ratioigf_cmean_value.csv"))[0])
    ferp_power_igf = pd.read_csv(list(pathlib.Path(data_dir).glob("erp_powerigf.csv"))[0])
    ferp_power_igf_cmean = pd.read_csv(list(pathlib.Path(data_dir).glob("erp_powerigf_cmean.csv"))[0])
    ferp_power_igf_value = pd.read_csv(list(pathlib.Path(data_dir).glob("erp_powerigf_value.csv"))[0])
    ferp_power_igf_value_cmean = pd.read_csv(list(pathlib.Path(data_dir).glob("erp_powerigf_cmean_value.csv"))[0])
    meta = pd.read_csv('../../data/raw/meta.csv')
    meta = meta.set_index('id').loc[ftotalpower_ratio_igf.subj]
    meta = meta.iloc[:,6:]
    channels = ['Fz', 'FCz', 'FC2', 'FC1', 'F2', 'F1', 'C2', 'C1', 'Cz']
    ftotalpower_ratio_igf_mean = ftotalpower_ratio_igf.set_index('subj')[channels].mean(axis=1)
    ftotalpower_ratio_igf_value_mean = ftotalpower_ratio_igf_value.set_index('subj')[channels].mean(axis=1)
    ferp_power_igf_mean = ferp_power_igf.set_index('subj')[channels].mean(axis=1)
    ferp_power_igf_value_mean = ferp_power_igf_value.set_index('subj')[channels].mean(axis=1)
    calc_cor(meta.copy(), 'total_power_ratio_igf', ftotalpower_ratio_igf_mean.copy(),saveDir)
    calc_cor(meta.copy(), 'total_power_ratio_igf_cmean', ftotalpower_ratio_igf_cmean.copy()['mean_center'].values,saveDir)
    calc_cor(meta.copy(), 'total_power_ratio_igf_value', ftotalpower_ratio_igf_value_mean.copy(),saveDir)
    calc_cor(meta.copy(), 'total_power_ratio_igf_cmean_value', ftotalpower_ratio_igf_value_cmean.copy()['mean_center'].values,saveDir)
    calc_cor(meta.copy(), 'erp_power_igf', ferp_power_igf_mean.copy(),saveDir)
    calc_cor(meta.copy(), 'erp_power_igf_cmean', ferp_power_igf_cmean.copy()['mean_center'].values,saveDir)
    calc_cor(meta.copy(), 'erp_power_igf_value', ferp_power_igf_value_mean.copy(),saveDir)
    calc_cor(meta.copy(), 'erp_power_igf_cmean_value', ferp_power_igf_value_cmean.copy()['mean_center'].values,saveDir)

def calc_cor(metadata, name, measure, saveDir):
    for col in metadata:
        r,p=stats.pearsonr(metadata[col].values,measure)
        csvfile=saveDir.joinpath(name+'.csv')
        if not csvfile.exists():
            csvfile.write_text('meta, measure, r,p\n')
        with open(csvfile,'a') as f:
                f.write(f'{col},{name},{r},{p}\n')
    metadata.loc[:,name] = measure
    meta_flat = metadata.melt(id_vars=name)
    fig=sns.lmplot(data=meta_flat,x=name,y='value',hue='variable')
    fig.savefig(saveDir.joinpath(name+'.png'))
    plt.close()

if __name__ == "__main__":
    igf_meta_corr()

