import re
import os
import itertools
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
sys.path.append(".")
from scipy.io import loadmat
os.chdir('./examples')
from UQPyL.utility.metrics import r_square

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
# #user setting
# # work_path="D:\SiHuRiver\model\FuTIanSi001\Scenarios\Test\TxtInOut"
# # para_file_name="paras_infos.txt"

# # begin_year=2009
# # begin_day=1
# # end_year=2016
# # end_day=366

# # begin_data=datetime(begin_year, 1, 1) + timedelta(begin_day - 1)
# # end_data=datetime(end_year, 1, 1) + timedelta(end_day - 1)
# # days_between=(end_data-begin_data).days+1
# # begin_days_calibration=(datetime(2011,1,1)-begin_data).days-1
# # end_days_calibration=(datetime(2014,12,31)-begin_data).days

# #initial 
# # paras_files = pd.read_csv('SWAT_paras_files.txt', sep=' ', names=['parameter', 'file'],  index_col='parameter')
# # HRU_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol"]
# # Watershed_suffiex=["pnd", "rte", "sub", "swq", "wgn", "wus"]

# # Watershed={}
# # with open(os.path.join(work_path, "fig.fig"), "r") as f:
# #     lines=f.readlines()
# #     for line in lines:
# #         match = re.search(r'(\d+)\.sub', line)
# #         if match:
# #             Watershed[match.group(1)]=[]

# # for sub in Watershed:
# #     file_name=sub+".sub"
# #     with open(os.path.join(work_path, file_name), "r") as f:
# #         lines=f.readlines()
# #         for line in lines:
# #             match = re.search(r'(\d+)\.mgt', line)
# #             if match:
# #                 Watershed[sub].append(match.group(1))
                
# # total_sub_list = list(itertools.chain.from_iterable(Watershed.values()))
# # watershed_list=list(Watershed.keys())

# # paras_infos=pd.read_csv(para_file_name, sep=' ', names=['Parameter', 'mode', 'low_bound', 'up_bound'],  index_col='Parameter')
# # low_bound= paras_infos['low_bound'].values
# # up_bound= paras_infos['up_bound'].values
# # paras_list=paras_infos.index.tolist()
# # num_paras=paras_infos.shape[0]
# # ##generate default_values
# # default_values=pd.DataFrame(columns=['para_name', 'file_name', 'value'])

# # for i in range(num_paras):
# #     para_name=paras_list[i]
# #     file_suffix=paras_files.loc[para_name, 'file']
    
# #     if file_suffix in HRU_suffix:
# #         for sub in total_sub_list:
# #             file_name=sub+"."+file_suffix
# #             file_path=os.path.join(work_path, file_name)
# #             if file_suffix=='sol':
# #                 default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras_for_sol(file_path, para_name)]
# #             else:
# #                 default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
# #     elif file_suffix in Watershed_suffiex:
# #         for sub in watershed_list:
# #             file_name=sub+"."+file_suffix
# #             file_path=os.path.join(work_path, file_name)
# #             default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
# #     elif file_suffix=="bsn":
# #         file_path=os.path.join(work_path, "basins."+file_suffix)
# #         default_values.loc[len(default_values)]=[para_name, "basins."+file_suffix, _get_default_paras(file_path, para_name)]

# # default_values.to_excel('default_values.xlsx', index=True)

# #setting values
# # paras_values=np.random.random((1, num_paras))*(up_bound-low_bound)+low_bound
# # paras_values=paras_values.ravel()
# # for i in range(num_paras):
# #     para_name=paras_list[i]
# #     mode=paras_infos.loc[para_name, 'mode']
# #     value=paras_values[i]
# #     file_suffix=paras_files.loc[para_name, 'file']
    
# #     if file_suffix in HRU_suffix:
# #         for sub in total_sub_list:
# #             file_name=sub+"."+file_suffix
# #             file_path=os.path.join(work_path, file_name)
# #             if file_suffix=='sol':
# #                 _set_paras_for_sol(file_path, para_name, value, mode, str.split(default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0]))
# #             else:
# #                 _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
# #     elif file_suffix in Watershed_suffiex:
# #         for sub in watershed_list:
# #             file_name=sub+"."+file_suffix
# #             file_path=os.path.join(work_path, sub+"."+file_suffix)
# #             _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
# #     elif file_suffix=="bsn":
# #         file_name="basins."+file_suffix
# #         file_path=os.path.join(work_path, file_name)
# #         _set_paras(file_path, para_name, value, mode, default_values.loc[(default_values['para_name'] == para_name) & (default_values['file_name'] == file_name), 'value'].values[0])
# #observe
# file_name="observed.txt"
# file_path=os.path.join(work_path, file_name)

# data=[]
# with open(file_path, "r") as f:
#     lines=f.readlines()
#     pattern = re.compile(r'(\d+)\s+FLOW_OUT_(\d+)_(\d+)\s+(\d+)')
#     for line in lines:
#         match = pattern.match(line)
#         if match:
#             index, day, year,  value = map(int, match.groups())
#             data.append([index, day, year,  value])
# observed_data = pd.DataFrame(data, columns=['index', 'day', 'year', 'value']).set_index('index')
# values_array = observed_data['value'].to_numpy()

# ############################evaluation##################

# def evaluate():
    
#     file_name="output.rch"
#     file_path=os.path.join(work_path, file_name)
#     begin_ind=0
#     with open(file_path, "r") as f:
#         lines=f.readlines()
#         for i, line in enumerate(lines):
#             match = re.search(r".*FLOW_OUTcms.*", line)
#             if match:
#                 begin_ind=i
#                 break
#     colspecs = [
#         (7, 11),  # RCH
#         (22, 26), # MON
#         (52, 62), # FLOW_OUTcms
#     ]
#     # name_list=['RCH', 'MON', 'FLOW_OUTcms']

#     df = pd.read_fwf(file_path, colspecs=colspecs, header=None, names=['RCH', 'MON', 'FLOW_OUTcms'], skiprows=begin_ind+1)
#     df_rch_40 = df.query('RCH==40')
#     flow_outcms_np = df_rch_40['FLOW_OUTcms'].to_numpy()
    
#     r2=r_square(values_array[begin_days_calibration:end_days_calibration], flow_outcms_np[begin_days_calibration:end_days_calibration])
    
#     return r2


###############################
from UQPyL.problems import Problem
from UQPyL.utility.metrics import r_square
from datetime import datetime, timedelta
import subprocess
class SWAT_CUP(Problem):
    HRU_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol"]
    Watershed_suffiex=["pnd", "rte", "sub", "swq", "wgn", "wus"]
    n_output=1
    def __init__(self, work_path, para_file_name, observe_file_name, swat_exe_path, rch_id):
        self.work_path=work_path
        self.para_file_name=para_file_name
        self.observe_file_name=observe_file_name
        self.exe_path=os.path.join(self.work_path, swat_exe_path)
        self.rch_id=rch_id
        
        self._initial()
        self._recond_default_values() #assign self.lb self.ub self.n_input
        self._get_observed_data()
    
    def evaluate(self, X):
        n=X.shape[0]
        Y=np.zeros((n,1))
        for i in range(n):
            self._set_values(X[i])
            try:
                subprocess.run(self.exe_path, cwd=self.work_path,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                simulation_data=self._get_simulation_data()
                rch_flows=simulation_data.query('RCH=={}'.format(self.rch_id))['FLOW_OUTcms'].to_numpy()
                Y[i,0]=-r_square(self.observed_data, rch_flows[self.begin_calibration-1:self.end_calibration])
            except:
                Y[i,0]=-np.inf
        return Y
     
    def _get_simulation_data(self):
        file_name="output.rch"
        file_path=os.path.join(self.work_path, file_name)
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
        # name_list=['RCH', 'MON', 'FLOW_OUTcms']

        simulation_data = pd.read_fwf(file_path, colspecs=colspecs, header=None, names=['RCH', 'MON', 'FLOW_OUTcms'], skiprows=begin_ind+1)
        
        return simulation_data
    
    def _get_observed_data(self):
        file_path=os.path.join(self.work_path, self.observe_file_name)
        data=[]
        with open(file_path, "r") as f:
            lines=f.readlines()
            pattern = re.compile(r'(\d+)\s+FLOW_OUT_(\d+)_(\d+)\s+(\d+\.?\d*)')
            for line in lines:
                match = pattern.match(line)
                if match:
                    index, day, year = map(int, match.groups()[:-1])
                    value = float(match.groups()[-1])
                    data.append([index, day, year,  value])
        observed_data = pd.DataFrame(data, columns=['index', 'day', 'year', 'value'])
        observed_data = observed_data.astype({'index': int, 'day': int, 'year': int, 'value': float}).set_index('index')
        values_array = observed_data['value'].to_numpy(dtype=float)
        
        first_record = observed_data.iloc[0]
        self.begin_calibration=(datetime(int(first_record['year']), 1, 1) + timedelta(int(first_record['day']) - 1)-self.begin_date).days
        
        last_record = observed_data.iloc[-1]
        self.end_calibration=(datetime(int(last_record['year']), 1, 1) + timedelta(int(last_record['day']) - 1)-self.begin_date).days
        
        self.observed_data=values_array
        
    def _set_values(self, paras_values):
        
        paras_values=paras_values.ravel()
        for i in range(self.n_input):
            para_name=self.paras_list[i]
            mode=self.paras_infos.loc[para_name, 'mode']
            value=paras_values[i]
            file_suffix=self.paras_files.loc[para_name, 'file']
            
            if file_suffix in self.HRU_suffix:
                for sub in self.total_sub_list:
                    file_name=sub+"."+file_suffix
                    file_path=os.path.join(self.work_path, file_name)
                    if file_suffix=='sol':
                        _set_paras_for_sol(file_path, para_name, value, mode, str.split(self.default_values.loc[(self.default_values['para_name'] == para_name) & (self.default_values['file_name'] == file_name), 'value'].values[0]))
                    else:
                        _set_paras(file_path, para_name, value, mode, self.default_values.loc[(self.default_values['para_name'] == para_name) & (self.default_values['file_name'] == file_name), 'value'].values[0])
            elif file_suffix in self.Watershed_suffiex:
                for sub in self.watershed_list:
                    file_name=sub+"."+file_suffix
                    file_path=os.path.join(self.work_path, sub+"."+file_suffix)
                    _set_paras(file_path, para_name, value, mode, self.default_values.loc[(self.default_values['para_name'] == para_name) & (self.default_values['file_name'] == file_name), 'value'].values[0])
            elif file_suffix=="bsn":
                file_name="basins."+file_suffix
                file_path=os.path.join(self.work_path, file_name)
                _set_paras(file_path, para_name, value, mode, self.default_values.loc[(self.default_values['para_name'] == para_name) & (self.default_values['file_name'] == file_name), 'value'].values[0])
            
    def _recond_default_values(self):
        
        para_file_name=os.path.join(self.work_path, self.para_file_name)
        paras_infos=pd.read_csv(para_file_name, sep=' ', names=['Parameter', 'mode', 'low_bound', 'up_bound'],  index_col='Parameter')
        self.lb= paras_infos['low_bound'].values
        self.ub= paras_infos['up_bound'].values
        paras_list=paras_infos.index.tolist()
        self.x_labels=paras_list
        num_paras=paras_infos.shape[0]
        self.n_input=num_paras
        ##generate default_values
        default_values_path=os.path.join(self.work_path, 'default_values.xlsx')
        if not os.path.exists(default_values_path):
            default_values=pd.DataFrame(columns=['para_name', 'file_name', 'value'])

            for i in range(num_paras):
                para_name=paras_list[i]
                file_suffix=self.paras_files.loc[para_name, 'file']
                
                if file_suffix in self.HRU_suffix:
                    for sub in self.total_sub_list:
                        file_name=sub+"."+file_suffix
                        file_path=os.path.join(self.work_path, file_name)
                        if file_suffix=='sol':
                            default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras_for_sol(file_path, para_name)]
                        else:
                            default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
                elif file_suffix in self.Watershed_suffiex:
                    for sub in self.watershed_list:
                        file_name=sub+"."+file_suffix
                        file_path=os.path.join(self.work_path, file_name)
                        default_values.loc[len(default_values)]=[para_name, file_name, _get_default_paras(file_path, para_name)]
                elif file_suffix=="bsn":
                    file_path=os.path.join(self.work_path, "basins."+file_suffix)
                    default_values.loc[len(default_values)]=[para_name, "basins."+file_suffix, _get_default_paras(file_path, para_name)]

            
            default_values.to_excel(default_values_path, index=True)
        else:
            default_values=pd.read_excel(default_values_path, index_col=0)
        self.default_values=default_values
        self.paras_infos=paras_infos
        self.paras_list=paras_list
    def _initial(self):
        Watershed={}
        
        NYSKIP=0
        #read control file fig.cio
        with open(os.path.join(self.work_path, "file.cio"), "r") as f:
            lines=f.readlines()
            for line in lines:
                match1 = re.search(r"(\s*)(\d+)(\s*.*NBYR.*)", line)
                if match1:
                    num_years=int(match1.group(2))
                
                match2 = re.search(r"(\s*)(\d+)(\s*.*IYR.*)", line)
                if match2:
                    begin_year=int(match2.group(2))
                
                match3 = re.search(r"(\s*)(\d+)(\s*.*IDAF.*)", line)
                if match3:
                    begin_day=int(match3.group(2))
                
                match4 = re.search(r"(\s*)(\d+)(\s*.*IDAL.*)", line)
                if match4:
                    end_day=int(match4.group(2))
                    
                match5 = re.search(r"(\s*)(\d+)(\s*.*NYSKIP.*)", line)
                if match5:
                    NYSKIP=int(match5.group(2))
                    
        self.begin_date=datetime(begin_year+NYSKIP, 1, 1) + timedelta(begin_day - 1)
        self.end_date=datetime(begin_year+num_years-1, 1, 1) + timedelta(end_day - 1)
        self.simulation_days=(self.end_date-self.begin_date).days+1
            
        #read control file fig.fig
        with open(os.path.join(self.work_path, "fig.fig"), "r") as f:
            lines=f.readlines()
            for line in lines:
                match = re.search(r'(\d+)\.sub', line)
                if match:
                    Watershed[match.group(1)]=[]
                    
        #read sub files   
        for sub in Watershed:
            file_name=sub+".sub"
            with open(os.path.join(self.work_path, file_name), "r") as f:
                lines=f.readlines()
                for line in lines:
                    match = re.search(r'(\d+)\.mgt', line)
                    if match:
                        Watershed[sub].append(match.group(1))
        
        self.Watershed=Watershed
        self.total_sub_list = list(itertools.chain.from_iterable(Watershed.values()))
        self.watershed_list=list(Watershed.keys())
        #read para_file
        file_path=os.path.join(self.work_path, 'SWAT_paras_files.txt')
        self.paras_files = pd.read_csv(file_path, sep=' ', names=['parameter', 'file'],  index_col='parameter')


swat_pro=SWAT_CUP(work_path="D:\SiHuRiver\model\FuTIanSi001\Scenarios\Test\TxtInOut",
                    para_file_name="paras_infos.txt",
                    observe_file_name="observed.txt",
                    swat_exe_path="swat_64rel.exe",
                    rch_id=40)
problem=swat_pro
print("################1.Sobol################")
from UQPyL.sensibility import Sobol
sobol_method=Sobol(problem=problem, cal_second_order=False) #Using Sobol Sequence and saltelli_sequence
X=sobol_method.sample(32)
Y=problem.evaluate(X)
Si=sobol_method.analyze(X, Y, verbose=True)

################2. FAST##################
print("################2.FAST################")
from UQPyL.sensibility import FAST
fast_method=FAST(problem=problem, M=4)
X=fast_method.sample(64)
Y=problem.evaluate(X)
Si=fast_method.analyze(X, Y, verbose=True)

print("#############4.Morris#############")
from UQPyL.sensibility import Morris
morris_method=Morris(problem=problem, num_levels=4) #Using Morris Sampler
X=morris_method.sample(64)
Y=problem.evaluate(X)
Si=morris_method.analyze(X, Y, verbose=True)

print("#############6.MARS_SA#############")
from UQPyL.sensibility import MARS_SA
mars_method=MARS_SA(problem=problem)
X=mars_method.sample(64*26)
Y=problem.evaluate(X)
Si=mars_method.analyze(X, Y, verbose=True)


# simulation_data=swat_pro._get_simulation_data()
# rch_flows=simulation_data.query('RCH=={}'.format(swat_pro.rch_id))['FLOW_OUTcms'].to_numpy()
# res=r_square(swat_pro.observed_data, rch_flows[swat_pro.begin_calibration-1:swat_pro.end_calibration])
# print(res)
# import matplotlib.pyplot as plt
# plt.plot(swat_pro.observed_data)
# plt.plot(rch_flows[swat_pro.begin_calibration-1:swat_pro.end_calibration])
# plt.show()



# from UQPyL.DoE import LHS
# from UQPyL.optimization import GA
# # lhs=LHS(criterion='classic', problem=swat_pro)
# # X=lhs.sample(50, swat_pro.n_input, random_seed=1)
# # Y=swat_pro.evaluate(X)
# ga=GA(problem=swat_pro,n_samples=50, maxFEs=1000000, maxIterTimes=10000)
# ga.run()

        






