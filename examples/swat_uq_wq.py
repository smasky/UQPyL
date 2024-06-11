from swat_utility import read_value_swat, copy_origin_to_tmp, write_value_to_file, read_simulation
import os
import re
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from datetime import datetime, timedelta
import subprocess
import queue
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json

import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')

from UQPyL.DoE import LHS
from UQPyL.problems import ProblemABC
from UQPyL.utility.metrics import r_square

class SWAT_UQ():
    hru_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol", "ops"]
    watershed_suffix=["pnd", "rte", "sub", "swq", "wgn", "wus"]
    n_output=3
    n_hru=0
    n_rch=0
    n_sub=0
    record_days=0
    observe_infos={}
    def __init__(self, work_path: str, paras_file_name: str, observed_file_name: str, 
                 swat_exe_name: str, special_paras_file: str=None,
                 max_workers: int=12, num_parallel: int=5):
        self.work_temp_dir=tempfile.mkdtemp()
        self.work_path=work_path
        self.paras_file_name=paras_file_name
        self.observed_file_name=observed_file_name
        self.swat_exe_name=swat_exe_name
        self.special_paras_file=special_paras_file
        
        self.max_workers=max_workers
        self.num_parallel=num_parallel
        
        self._initial()
        self._record_default_values()
        # self._get_observed_data()
        
        self.work_path_queue=queue.Queue()
        self.work_temp_dirs=[]
        
        for i in range(num_parallel):
            path=os.path.join(self.work_temp_dir, "instance{}".format(i))
            self.work_temp_dirs.append(path)
            self.work_path_queue.put(path)
        
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [executor.submit(copy_origin_to_tmp, self.work_path, work_temp) for work_temp in self.work_temp_dirs]
        for future in futures:
            future.result()
    #------------------------------------------interface function-------------------------------------------------#
    def evaluate(self, X):
        """
        evaluate the objective within X
        """
        n=X.shape[0]
        Y=np.zeros((n,3))
        
        self._subprocess(X[0, :], 0)
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures=[executor.submit(self._subprocess, X[i, :], i) for i in range(n)]

        for i, future in enumerate(futures):
            TN, TP=future.result()
            Y[i, 0]=TN[0]
            Y[i, 1]=TP[0]
            Y[i, 2]= self._cost(X[i, :])
            
        return Y
    
    #------------------------------------private function-----------------------------------------------#
    def _discrete_variable_transform(self, X):
        lb=self.lb.ravel()
        ub=self.ub.ravel()
        if self.disc_var is not None:
            for i in range(self.n_input):
                if self.disc_var[i]==1:
                    num_interval=len(self.disc_range[i])
                    bins=np.linspace(lb[i], ub[i], num_interval+1)
                    indices = np.digitize(X[:, i], bins[1:])
                    indices[indices >= num_interval] = num_interval - 1
                    if isinstance(self.disc_range[i], list):
                        X[:, i]=np.array(self.disc_range[i])[indices]
                    else:
                        X[:, i]=self.disc_range[i][indices]
        return X
    def _subprocess(self, input_x, id):
        """
        subprocess for run swat with each input_X
        """
        work_path=self.work_path_queue.get()
        self._set_values(work_path, input_x)
        process= subprocess.Popen(
            os.path.join(work_path, self.swat_exe_name),
            cwd=work_path,
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True)
        process.wait()
        TN=read_simulation(os.path.join(work_path, "output.rch"), 48, 51, 62, 1560, 1622)
        TP=read_simulation(os.path.join(work_path, "output.rch"), 49, 51, 62, 1560, 1622)
        self.work_path_queue.put(work_path)
        return (TN, TP)
    
    def _cost(self, input_x):
        
        cost=0
        for i, j in enumerate(range(0,10,2)):
            cost+=420*(input_x[j]*input_x[j+1])*input_x[20+i]
        weight=[15248, 30259, 7253, 20759, 32553]
        for i, j in enumerate(range(10,20,2)):
            cost+=6000*weight[i]/input_x[j]*input_x[25+i]*input_x[j+1]
        return cost
    
    def _set_values(self, work_path, paras_values):
        """
        set_value
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures=[]
            for file_name, infos in self.file_var_info.items():
                future = executor.submit(write_value_to_file, work_path, file_name, 
                                         infos["name"], infos["default"], 
                                         infos["index"], infos["mode"],  infos["position"], infos["type"],
                                         paras_values.ravel())
                futures.append(future)
            
            for future in futures:
                res=future.result()
                  
    def _record_default_values(self):

        var_infos_path=os.path.join(self.work_path, self.paras_file_name)
        low_bound=[]
        up_bound=[]
        disc_var=[]
        var_name=[]
        mode=[]
        assign_range=[]
        value_range=[]
        with open(var_infos_path, 'r') as f:
            lines=f.readlines()
            for line in lines:
                tmp_list=line.split()
                var_name.append(tmp_list[0])
                mode.append(tmp_list[1])
                op_type=tmp_list[2]
                op_tmp=tmp_list[3].split("_")
                
                if op_type=="c":
                    low_bound.append(float(op_tmp[0]))
                    up_bound.append(float(op_tmp[1]))
                    value_range.append(0)
                    disc_var.append(0)
                else:
                    low_bound.append(float(op_tmp[0]))
                    up_bound.append(float(op_tmp[-1]))
                    value_range.append([float(e) for e in op_tmp])
                    disc_var.append(1)
                    
                assign_range.append(tmp_list[4:])
                
        self.lb= np.array(low_bound).reshape(1,-1)
        self.ub= np.array(up_bound).reshape(1,-1)
        self.mode= mode
        self.paras_list=var_name
        self.x_labels=self.paras_list
        self.disc_range=value_range
        self.disc_var=disc_var
        self.n_input=len(self.paras_list)
        
        self.file_var_info={}
        
        self.data_types=[]
        for i, element in enumerate(self.paras_list):
            element=element.split('@')[0]
            suffix=self.paras_file.query('para_name==@element')['file_name'].values[0]
            position=self.paras_file.query('para_name==@element')['position'].values[0]
            
            if(self.paras_file.query('para_name==@element')['type'].values[0]=="int"):
                data_type_=0
            else:
                data_type_=1
            self.data_types.append(data_type_)
            if suffix in self.hru_suffix:
                if assign_range[i][0]=="all":
                    files=[e+".{}".format(suffix) for e in self.hru_list]
                else:
                    files=[]
                    for ele in assign_range[i]:
                        if "_" not in ele:
                            code=f"{'0' * (9 - 4 - len(ele))}{ele}{'0'*4}"
                            for e in self.watershed_hru[code]:
                                files.append(e+"."+suffix)
                        else:
                            bsn_id, hru_id=ele.split('_')
                            code=f"{'0' * (9 - 4 - len(bsn_id))}{bsn_id}{'0'*(4-len(hru_id))}{bsn_id}"
                            files.append(code+"."+suffix)
            elif suffix in self.watershed_suffix:
                if assign_range[i][0]=="all":
                    files=[e+"."+suffix for e in self.watershed_list]
                else:
                    files=[e+"."+suffix for e in assign_range[i]]
            elif suffix=="bsn":
                files=["basins.bsn"]
            
            for file in files:
                self.file_var_info.setdefault(file,{})
                self.file_var_info[file].setdefault("index", [])
                self.file_var_info[file]["index"].append(i)
                self.file_var_info[file].setdefault("mode", [])
                if self.mode[i]=="v":
                    self.file_var_info[file]["mode"].append(0)
                elif self.mode[i]=="r":
                    self.file_var_info[file]["mode"].append(1)
                elif self.mode[i]=="a":
                    self.file_var_info[file]["mode"].append(2)
                
                self.file_var_info[file].setdefault("name", [])
                self.file_var_info[file]["name"].append(element)
                self.file_var_info[file].setdefault("position",[])
                self.file_var_info[file]["position"].append(position)
                self.file_var_info[file].setdefault("type", [])
                self.file_var_info[file]["type"].append(data_type_)
                                
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务到线程池
            futures=[]
            for file_name, infos in self.file_var_info.items():
                futures.append(executor.submit(read_value_swat, self.work_path, file_name , infos["name"], infos["position"], 1))

        for future in futures:
            res=future.result()
            for key, items in res.items():
                values=' '.join(str(value) for value in items)
                _, file_name=key.split('|')
                self.file_var_info[file_name].setdefault("default", [])
                self.file_var_info[file_name]["default"].append(values)
                          
    def delete(self):
        shutil.rmtree(self.work_temp_dir)
            
    def _initial(self):
 
        #read control file fig.cio
        paras=["NBYR", "IYR", "IDAF", "IDAL", "NYSKIP"]
        pos=["default"]*5
        dict_values=read_value_swat(self.work_path, "file.cio", paras, pos, 0)
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
        
        self.watershed_hru=watershed
        self.hru_list = list(itertools.chain.from_iterable(watershed.values()))
        self.watershed_list=list(watershed.keys())
        
        self.n_hru=len(self.hru_list)
        self.n_sub=len(self.watershed_list)
        self.n_rch=self.n_sub
        self.record_days=self.simulation_days-self.output_skip_days
        
        self.paras_file=pd.read_excel(os.path.join(self.work_path, 'SWAT_paras_files.xlsx'), index_col=0)
        if self.special_paras_file is not None:
            with open(os.path.join(self.work_path, self.special_paras_file), 'r') as f:
                lines=f.readlines()
                for line in lines:
                    tmp_list=line.split()
                    self.paras_file.loc[tmp_list[0]]=tmp_list[1:]
        
file_path="D:\SWAT\TxtInOut"
from UQPyL.DoE import LHS    
swat_cup=SWAT_UQ(work_path=file_path,
                    paras_file_name="paras_infos.txt",
                    observed_file_name="observed.txt",
                    swat_exe_name="swat_681.exe",
                    special_paras_file="special_paras.txt",
                    max_workers=10, num_parallel=5)
from UQPyL.optimization import NSGAII
nsga=NSGAII(swat_cup, n_samples=100)
nsga.run()