import pathlib
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import seaborn as sns
import pickle

def plot_psds():
    data_dir = "../../data/interim/powers"
    failai = list(pathlib.Path(data_dir).glob("*.csv"))
    saveDir = pathlib.Path("../../reports/figures/powers")
    saveDir.mkdir(exist_ok=True)
    saveDirData = pathlib.Path("../../data/interim/igf")
    saveDirData.mkdir(exist_ok=True)
    for ifile in failai:
        #  ifile = failai[0]
        df = pd.read_csv(ifile)
        df=df.dropna()
        chanels = ['Fz', 'FCz', 'FC2', 'FC1', 'F2', 'F1', 'C2', 'C1', 'Cz']
        datt=df[df['ichan'].isin(chanels)]
        dff = datt.pivot_table(values='spectra',index=['ifreq'],columns='stimfreq')
        plt.figure(figsize=(20,10))
        ax=plt.axes()
        dff.plot(ax=ax,label='GA')
        plt.xlabel('frequency, Hz')
        plt.tight_layout()
        plt.savefig(saveDir.joinpath(f'{ifile.stem}_GA.png'))
        plt.close('all')
        for group, datt in df.groupby(by=['subjname']):
            dattch=datt[datt['ichan'].isin(chanels)]
            dff = dattch.pivot_table(values='spectra',index=['ifreq'],columns='stimfreq')
            fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(20,10))
            dff.plot(ax=ax[0][0],label=f'{group}')
            ax[0][0].set_xlabel('frequency, Hz')
            ax[0][0].legend(ncol=3,fontsize=6,columnspacing=1)
            profile=dattch[dattch.stimfreq==dattch.ifreq].pivot_table(values='spectra',index='ifreq')
            profile.plot(ax=ax[0][1])
            ax[0][1].set_xlabel('frequency, Hz')
            ax[0][1].legend().remove()
            ax[0][1].set_title(f'{ifile.stem} {group}')
            profile_max=profile.max()[0]
            profile_idxmax=profile.idxmax()[0]
            profile_values_mean = pd.DataFrame({'igf':[profile_idxmax],'igf_value':[profile_max]})
            ax[0][1].text(profile_idxmax,profile_max-.1,f'{int(profile_idxmax)} Hz\n {profile_max:.2}')
            profile=datt[datt.stimfreq==datt.ifreq].pivot_table(values='spectra',index=['ifreq'], columns='ichan')
            profile_values = pd.DataFrame({'igf':profile.idxmax(),'igf_value':profile.max()})
            chnames = pickle.load(open('chnames.p','rb'))
            pos=pickle.load(open('info.p','rb'))
            im=mne.viz.plot_topomap(profile_values.igf,pos,axes=ax[1][0],cmap=cm.viridis,contours=0,vmin=35,show=False)
            plt.colorbar(im[0],ax=ax[1][0])
            im=mne.viz.plot_topomap(profile_values.igf_value,pos,cmap=cm.viridis,
                    contours=0, axes=ax[1][1],show=False)
            plt.colorbar(im[0],ax=ax[1][1])
            plt.tight_layout()
            plt.savefig(saveDir.joinpath(f'{ifile.stem}_{group}.png'))
            plt.close('all')
            save_igf(profile_values.igf, ifile.stem, 'igf', group, saveDirData)
            save_igf_mean(profile_values_mean.igf, ifile.stem, 'igf_cmean', group, saveDirData)
            save_igf(profile_values.igf_value, ifile.stem, 'igf_value', group, saveDirData)
            save_igf_mean(profile_values_mean.igf_value, ifile.stem, 'igf_cmean_value', group, saveDirData)

def save_igf(values,  powertype, datatype, subj, saveDir):
    csvfile = saveDir.joinpath(powertype + datatype+'.csv')
    if not pathlib.Path(csvfile).exists():
        nam = [x for x in values.index]
        nam = ','.join(nam)
        csvfile.write_text(f'subj,'+nam+'\n')
    with open(csvfile,'a') as f:
        vt = [str(x) for x in values.values]
        vt = ','.join(vt)
        f.write(f"{subj+','+vt}"+'\n')

def save_igf_mean(values,  powertype, datatype, subj, saveDir):
    csvfile = saveDir.joinpath(powertype + datatype+'.csv')
    if not pathlib.Path(csvfile).exists():
        csvfile.write_text(f'subj,'+'mean_center'+'\n')
    with open(csvfile,'a') as f:
        vt = [str(x) for x in values.values]
        vt = ','.join(vt)
        f.write(f"{subj+','+vt}"+'\n')



if __name__ == "__main__":
    plot_psds()
