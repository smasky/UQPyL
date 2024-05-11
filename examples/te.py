import re
import os
import itertools
import pandas as pd
import numpy as np
os.chdir('./examples')

def _get_default_paras(file_path, name):
    pattern=r"(\s*)(\-?\d+\.?\d*)(\s*.*{}.*)".format(name)
    with open(file_path, "r+") as f:
        lines=f.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                return float(match.group(2))

def _get_default_paras_for_sol(file_path, name):
    pattern=r".*{}.*".format(name)
    with open(file_path, "r+") as f:
        lines=f.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                numbers=re.findall(r"\d+\.\d+", line)
                return " ".join(numbers)

def _set_paras_for_sol(file_path, name, value, mode, origin_value=None):
    pattern=r"\s*{}.*".format(name)
    with open(file_path, "r+") as f:
        lines=f.readlines()
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                numbers=re.findall(r"\d+\.\d+", line)
                for i, number in enumerate(numbers):
                    if mode=='r':
                        value=float(origin_value[i])*(1+value)
                    elif mode=='a':
                        value=float(origin_value[i])+value
                    line=line.replace(str(number), "{:.2f}".format(value))
                    lines[i]=line
                    break
        f.seek(0)
        f.writelines(lines)
        f.truncate()
                
def _set_paras(file_path, name, value, mode, origin_value=None):
    
    pattern=r"(\s*)(\-?\d+\.?\d*)(\s*.*{}.*)".format(name)
    with open(file_path, "r+") as f:
        lines=f.readlines()
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            
            if match:
                if mode=='r':
                    value=origin_value*(1+value)
                elif mode=='a':
                    value=origin_value+value
                    
                new_text = re.sub(pattern,  match.group(1)+"{}".format(value)+match.group(3), line)
                lines[i]=new_text
                # origin_value=match.group(2)    
                # if re.fullmatch(r'\-?\d+', origin_value) is not None and re.fullmatch(r'\-?\d+', str(value)) is not None: 
                #     if mode=='r':
                #         value=origin_value*(1+value)
                #     new_text = re.sub(pattern,  match.group(1)+"{}".format(value)+match.group(3), line)
                #     lines[i]=new_text
                # else:
                #     raise ValueError("Please check the type of input value for {} in {}.".format(name, file_path))
                break
        f.seek(0)
        f.writelines(lines)
        f.truncate()

#basic setting
paras_files = pd.read_csv('SWAT_paras_files.txt', sep=' ', names=['parameter', 'file'],  index_col='parameter')


work_path="H:\SiHuRiver\model\FuTIanSi001\Scenarios\Test\TxtInOut"
HRU_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol"]
Watershed_suffiex=["pnd", "rte", "sub", "swq", "wgn", "wus"]

Watershed={}
with open(os.path.join(work_path, "fig.fig"), "r") as f:
    lines=f.readlines()
    for line in lines:
        match = re.search(r'(\d+)\.sub', line)
        if match:
            Watershed[match.group(1)]=[]

for sub in Watershed:
    file_name=sub+".sub"
    with open(os.path.join(work_path, file_name), "r") as f:
        lines=f.readlines()
        for line in lines:
            match = re.search(r'(\d+)\.mgt', line)
            if match:
                Watershed[sub].append(match.group(1))
                
total_sub_list = list(itertools.chain.from_iterable(Watershed.values()))
watershed_list=list(Watershed.keys())

paras_infos=pd.read_csv('paras_infos.txt', sep=' ', names=['Parameter', 'mode', 'low_bound', 'up_bound'],  index_col='Parameter')
low_bound= paras_infos['low_bound'].values
up_bound= paras_infos['up_bound'].values
paras_list=paras_infos.index.tolist()
num_paras=paras_infos.shape[0]

paras_values=np.random.random((1, num_paras))*(up_bound-low_bound)+low_bound
paras_values=paras_values.ravel()

default_values=pd.DataFrame(columns=['para_name', 'file_name', 'value'])

# for i in range(num_paras):
#     para_name=paras_list[i]
#     file_suffix=paras_files.loc[para_name, 'file']
    
#     if file_suffix in HRU_suffix:
#         for sub in total_sub_list:
#             file_name=sub+"."+file_suffix
#             file_path=os.path.join(work_path, file_name)
#             if file_suffix=='sol':
#                 default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras_for_sol(file_path, para_name)]
#             else:
#                 default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
#     elif file_suffix in Watershed_suffiex:
#         for sub in watershed_list:
#             file_name=sub+"."+file_suffix
#             file_path=os.path.join(work_path, file_name)
#             default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
#     elif file_suffix=="bsn":
#         file_path=os.path.join(work_path, "basins."+file_suffix)
#         default_values.loc[len(default_values)]=[para_name, "basins."+file_suffix, _get_default_paras(file_path, para_name)]

# default_values.to_excel('default_values.xlsx', index=True)

# for i in range(num_paras):
#     para_name=paras_list[i]
#     mode=paras_infos.loc[para_name, 'mode']
#     value=paras_values[i]
#     file_suffix=paras_files.loc[para_name, 'file']
    
#     if file_suffix in HRU_suffix:
#         for sub in total_sub_list:
#             file_name=sub+"."+file_suffix
#             file_path=os.path.join(work_path, file_name)
#             if file_suffix=='sol':
#                 _set_paras_for_sol(file_path, para_name, value, mode, str.split(default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0]))
#             else:
#                 _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
#     elif file_suffix in Watershed_suffiex:
#         for sub in watershed_list:
#             file_name=sub+"."+file_suffix
#             file_path=os.path.join(work_path, sub+"."+file_suffix)
#             _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
#     elif file_suffix=="bsn":
#         file_name="basins."+file_suffix
#         file_path=os.path.join(work_path, file_name)
#         _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
        
############################evaluation##################
file_name="output.rch"
file_path=os.path.join(work_path, file_name)
begin_ind=0
with open(file_path, "r") as f:
    lines=f.readlines()
    for i, line in enumerate(lines):
        match = re.search(r".*FLOW_OUTcms.*", line)
        if match:
            begin_ind=i
            break
colspecs = [
    (7, 11),  # RCH
    (22, 26), # MON
    (52, 62), # FLOW_OUTcms
]
name_list=['RCH', 'MON', 'FLOW_OUTcms']

df = pd.read_fwf(file_path, colspecs=colspecs, header=None, names=['RCH', 'MON', 'FLOW_OUTcms'], skiprows=begin_ind+1)
df_rch_40 = df.query('RCH == 40')
flow_outcms_np = df_rch_40['FLOW_OUTcms'].to_numpy()

