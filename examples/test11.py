import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')

from my_module import read_value_swat, set_value_swat, read_simulation, copy_origin_to_tmp
import os
import re
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from UQPyL.DoE import LHS
from UQPyL.problems import Problem
from UQPyL.utility.metrics import r_square
from datetime import datetime, timedelta
import subprocess
import multiprocessing
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class SWAT_UQ():
    hru_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol"]
    watershed_suffix=["pnd", "rte", "sub", "swq", "wgn", "wus"]
    n_output=1
    def __init__(self, work_path: str, paras_file_name: str, observed_file_name: str, swat_exe_name: str, max_workers: int=12, num_parallel: int=5):
        
        self.work_temp_dir=tempfile.mkdtemp()
        self.work_path=work_path
        self.paras_file_name=paras_file_name
        self.observed_file_name=observed_file_name
        self.swat_exe_name=swat_exe_name
        
        self.max_workers=max_workers
        self.num_parallel=num_parallel
        
        self._initial()
        self._recond_default_values()
        self._get_observed_data()
        
        self.work_temp_dirs=[os.path.join(self.work_temp_dir, "instance{}".format(i)) for i in range(num_parallel)]
        
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [executor.submit(copy_origin_to_tmp, self.work_path, work_temp) for work_temp in self.work_temp_dirs]
        for future in futures:
            future.result()
        # for work_temp in self.work_temp_dirs:
        #     shutil.copytree(self.work_path, work_temp)
    
    def _subprocess(self, work_path, input_x, id):
        # try:
        a=time.time()
        self._set_values(work_path, input_x)
        subprocess.run(os.path.join(work_path, self.swat_exe_name), cwd=work_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rch_flows=np.array(read_simulation(os.path.join(work_path, "output.rch"), 6, 40, 41, 10))
        # rch_flows=simulation_data[self.rch_id-1::41]
        # rch_flows=simulation_data.query('RCH=={}'.format(self.rch_id))['FLOW_OUTcms'].to_numpy()
        y=-r_square(self.observed_data, rch_flows[self.begin_calibration-self.output_skip_days-1:self.end_calibration-self.output_skip_days])
        # except:
            # y=-np.inf 
        b=time.time()
        return (id, work_path, b-a)
    def evaluate(self, X):
        n=X.shape[0]
        Y=np.zeros((n,1))
        
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures=[executor.submit(self._subprocess, self.work_temp_dirs[i%self.num_parallel], X[i, :], i) for i in range(n)]

        for future in futures:
            res=future.result()
            print(res)
        return Y
    
    def _set_values(self, work_path, paras_values):
        
        paras_values=paras_values.ravel()
        
        tasks={}
        for key, item in self.tasks.items():
            para_list=item["para_list"]
            tasks.setdefault(key, [])
            for para in para_list:
                value=paras_values[self.dict_index_paras[para][0]]
                mode=self.dict_index_paras[para][1]
                tasks[key].append((para, value, mode))
                
        jobs={} 
        for key, items in tasks.items():
            if key in self.hru_suffix:
                for code in self.hru_list:
                    file_name=code+"."+key
                    jobs.setdefault(file_name, {})
                    for para, value, mode in items:
                        set_value=self._generate_value(file_name, para, value, mode)
                        jobs[file_name][para]=set_value
            elif key in self.watershed_suffix:
                for code in self.watershed_list:
                    file_name=code+"."+key
                    jobs.setdefault(file_name, {})
                    for para, value, mode in items:
                        set_value=self._generate_value(file_name, para, value, mode)
                        jobs[file_name][para]=set_value
            elif key in ['bsn']:
                file_name="basins."+key
                jobs.setdefault(file_name, {})
                for para, value, mode in items:
                    set_value=self._generate_value(file_name, para, value, mode)
                    jobs[file_name][para]=set_value                
                        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务到线程池
            futures=[]
            for key, item in jobs.items():
                futures.append(executor.submit(set_value_swat, os.path.join(work_path, key), list(item.keys()), item))
            for future in futures:
                future.result()
                   
    def _generate_value(self, key, para, value, mode):
        if mode=='v':
            str_value='{:.4f}'.format(value)
        elif mode=='r':
            str_origin_value=self.default_values.loc[(self.default_values['para_name'] == para) & (self.default_values['file_name'] == key), 'value'].values[0]
            if key[-3:]=='sol':
                numbers=str_origin_value.split()
                for i in range(len(numbers)):
                    numbers[i]=float(numbers[i])*(1+value)
                str_value = (' ' * 8).join(f'{num:.4f}' for num in numbers)
            else:
                str_value='{:.4f}'.format(float(str_origin_value)*(1+value))
        elif mode=='a':
            str_origin_value=self.default_values.loc[(self.default_values['para_name'] == para) & (self.default_values['file_name'] == key), 'value'].values[0]
            if key[-3:]=='sol':
                numbers=str_origin_value.split()
                for i in range(len(numbers)):
                    numbers[i]=float(numbers[i])+value
                str_value = '        '.join(f'{num:.4f}' for num in numbers)
            else:
                str_value='{:.4f}'.format(float(str_origin_value)+value)
        return str_value
    
    def _get_observed_data(self):
        file_path=os.path.join(self.work_path, self.observed_file_name)
        data=[]
        rch_id=0
        with open(file_path, "r") as f:
            lines=f.readlines()
            pattern1=re.compile(r'FLOW_OUT_(\d+)\s+')
            pattern = re.compile(r'(\d+)\s+FLOW_OUT_(\d+)_(\d+)\s+(\d+\.?\d*)')
            for line in lines:
                match1= pattern1.match(line)
                if rch_id==0 and match1:
                    rch_id=int(match1.group(1))
                    self.rch_id=rch_id
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
        
    def _recond_default_values(self):
        """
        
        """
        paras_infos=pd.read_csv(os.path.join(self.work_path, self.paras_file_name), sep=' ', names=['Parameter', 'mode', 'low_bound', 'up_bound'],  index_col='Parameter')
        self.lb= paras_infos['low_bound'].values
        self.ub= paras_infos['up_bound'].values
        self.mode=paras_infos['mode'].values
        self.paras_list=paras_infos.index.tolist()
        self.x_labels=self.paras_list
        
        self.n_input=len(self.paras_list)
        
        self.dict_index_paras={}
        for i, element in enumerate(self.paras_list):
            self.dict_index_paras[element]=(i, self.mode[i])
            
        ##generate default value database for all parameters
        default_database_path=os.path.join(self.work_path, 'default_values.xlsx')
        tasks={}
        # if not os.path.exists(default_database_path):
            
        default_values=pd.DataFrame(columns=['para_name', 'file_name', 'value'])
        
        for i in range(self.n_input):
            para_name=self.paras_list[i]
        
            try:
                type, suffix=self.paras_file.loc[para_name, ["type", "file_name"]]
            
            except KeyError:
                raise KeyError("The parameter is not in swat file, please check Swat_para_files.xlsx!")

            tasks.setdefault(suffix, {"para_list":[]})
            tasks[suffix]["para_list"].append(para_name)
        
        self.tasks=tasks
        
        jobs=[]
        for key, value in tasks.items():
            if key in self.hru_suffix:
                for code in self.hru_list:
                    jobs.append((self.work_path, code+"."+key, value["para_list"], 1))
                    # read_value_swat(os.path.join(self.work_path, code+"."+key), value["para_list"])
            elif key in self.watershed_suffix:
                for code in self.watershed_list:
                    jobs.append((self.work_path, code+"."+key, value["para_list"], 1))
            elif key in ['bsn']:
                jobs.append((self.work_path, "basins.bsn", value["para_list"], 1))
        
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务到线程池
            futures = [executor.submit(read_value_swat, *job) for job in jobs]

        for future in futures:
            res=future.result()
            for key, items in res.items():
                values=' '.join(str(value) for value in items)
                name, file_name=key.split('|')
                    # read_value_swat(os.path.join(self.work_path, code+"."+key), value["para_list"])
                default_values.loc[len(default_values)]=[name, file_name, values]
        self.default_values=default_values
        default_values.to_excel(default_database_path, index=True)
        
    def __del__(self):
        shutil.rmtree(self.work_temp_dir)
         
    def _initial(self):
        """
        
        """   
        #read control file fig.cio
        paras=["NBYR", "IYR", "IDAF", "IDAL", "NYSKIP"]
        dict_values=read_value_swat(self.work_path, "file.cio", paras, 0)
        self.begin_date=datetime(int(dict_values["IYR"][0]), 1, 1)+timedelta(int(dict_values['IDAF'][0])-1)
        self.end_date=datetime(int(dict_values["IYR"][0])+int(dict_values['NBYR'][0])-1, 1, 1)+timedelta(int(dict_values['IDAL'][0])-1)
        self.simulation_days=(self.end_date-self.begin_date).days+1
        output_skip_years=int(dict_values["NYSKIP"][0])
        self.output_skip_days=(datetime(int(dict_values["IYR"][0])+output_skip_years, 1, 1)+timedelta(int(dict_values['IDAF'][0])-1)-self.begin_date).days
        
        #read control file fig.fig
        watershed={}
        with open(os.path.join(self.work_path, "fig.fig"), "r") as f:
            lines=f.readlines()
            for line in lines:
                match = re.search(r'(\d+)\.sub', line)
                if match:
                    watershed[match.group(1)]=[]
        
        #read sub files
        for sub in watershed:
            file_name=sub+".sub"
            with open(os.path.join(self.work_path, file_name), "r") as f:
                lines=f.readlines()
                for line in lines:
                    match = re.search(r'(\d+)\.mgt', line)
                    if match:
                        watershed[sub].append(match.group(1))
        
        self.Watershed=watershed
        self.hru_list = list(itertools.chain.from_iterable(watershed.values()))
        self.watershed_list=list(watershed.keys())
        
        # self.paras_files = pd.read_csv(os.path.join(self.work_path, 'SWAT_paras_files.txt'), sep=' ', names=['parameter', 'file'],  index_col='parameter')
        self.paras_file=pd.read_excel(os.path.join(self.work_path, 'SWAT_paras_files.xlsx'), index_col=0)
        
    
file_path="D:\SiHuRiver\model\FuTIanSi001\Scenarios\Test1\TxtInOut"
# a=read_value_swat(file_path,file_name, ['Ave'], 0)
from UQPyL.DoE import LHS    
swat_cup=SWAT_UQ(work_path=file_path,
                    paras_file_name="paras_infos.txt",
                    observed_file_name="observed.txt",
                    swat_exe_name="swat_64rel.exe",
                    max_workers=6, num_parallel=6)
# rch_flows=np.array(read_simulation(os.path.join(file_path, "output.rch"), 6, 40, 41, 10))
# y=-r2_score(swat_cup.observed_data, rch_flows[swat_cup.begin_calibration-swat_cup.output_skip_days-1:swat_cup.end_calibration-swat_cup.output_skip_days])
# a=1
# from UQPyL.optimization import PSO
# pso=PSO(problem=swat_cup)
# pso.run()
lhs=LHS('classic')
X=lhs.sample(50, swat_cup.n_input)
X=X*(swat_cup.ub-swat_cup.lb)+swat_cup.lb
print(X)
# X=np.loadtxt('./pso.txt')
import time
a=time.time()
swat_cup.evaluate(X)
# for i in range(10):   
#     y=swat_cup._subprocess(file_path, X[-1, :], id=0)
#     print(y)

b=time.time()
print(b-a)
# for i in range(10):   
#     y=swat_cup._subprocess(file_path, X[i, :], id=0)
#     print(y)

        