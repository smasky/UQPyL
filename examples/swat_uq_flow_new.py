import sys
sys.path.append(".")

import os
import re
import queue
import shutil
import tempfile
import itertools
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from UQPyL.utility.metrics import r_square
from UQPyL.problems import ProblemABC

#C++ Module
from swat_utility import read_value_swat, copy_origin_to_tmp, write_value_to_file, read_simulation

class SWAT_UQ_Flow(ProblemABC):
    hru_suffix=["chm", "gw", "hru", "mgt", "sdr", "sep", "sol"]
    watershed_suffix=["pnd", "rte", "sub", "swq", "wgn", "wus"]
    model_infos={}
    observe_infos={}
    n_output=1
    n_hru=0
    n_rch=0
    n_sub=0
    def __init__(self, work_path: str, paras_file_name: str, 
                 observed_file_name: str, swat_exe_name: str, temp_path:str=None, 
                 max_threads: int=12, num_parallel: int=5):
        
        #create the space for running multiple instance of SWAT
        if temp_path is None:
            #if dont set the temp_path, create a temp dir
            self.work_temp_dir=tempfile.mkdtemp()
            self.use_temp_dir=True
        else:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            os.makedirs(temp_path)
            self.work_temp_dir=temp_path
            self.use_temp_dir=False
        
        #basic setting
        self.work_path=work_path
        self.paras_file_name=paras_file_name
        self.observed_file_name=observed_file_name
        self.swat_exe_name=swat_exe_name
        
        self.max_workers=max_threads
        self.num_parallel=num_parallel
        
        self._initial()
        self._record_default_values()
        self._get_observed_data()
        
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
                       
        super().__init__(n_input=len(self.paras_list), n_output=self.n_output, lb=self.lb, ub=self.ub, disc_var=[0]*len(self.paras_list), disc_range=[0]*len(self.paras_list))
    #------------------------interface function-----------------------#
    def _subprocess(self, input_x, id):
        
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
        
        rch_ids=self.observe_infos["rch_ids"]
        NSE=[]
        for rch_id in rch_ids:
            index_groups, true_flows=self.observe_infos["rch_flows"][rch_id]
            flows=[]
            for group in index_groups:
                startline=group[0]*self.n_rch+10
                endline=(group[1]+1)*self.n_rch+9
                flow=np.array(read_simulation(os.path.join(work_path, "output.rch"), 6, rch_id, self.n_rch, startline, endline))
                flows.append(flow)
            sim_flows=np.concatenate(flows, axis=0)
            nse=r_square(true_flows, sim_flows)
            NSE.append(nse)
        
        self.work_path_queue.put(work_path)
        return (id, np.mean(NSE))
    
    def evaluate(self, X):
        n=X.shape[0]
        Y=np.zeros((n,1))
        
        for i in range(n):
            id, nse=self._subprocess(X[i,:], i)
            Y[id, :]=-1*nse
        
        if n<self.num_parallel:
            for i in range(n):
                id, nse=self._subprocess(X[i,:], i)
                Y[id, :]=-1*nse
        else:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                futures=[executor.submit(self._subprocess, X[i, :], i) for i in range(n)]
            
            for i, future in enumerate(futures):
                 id, nse=future.result()
                 Y[id, :]=-1*nse
                  
        return Y
    #---------------------private function------------------------------#
    def _initial(self):
        """
        
        """   
        paras=["IPRINT", "NBYR", "IYR", "IDAF", "IDAL", "NYSKIP"]
        pos=["default"]*len(paras)
        dict_values=read_value_swat(self.work_path, "file.cio", paras, pos, 0)
        begin_date=datetime(int(dict_values["IYR"][0]), 1, 1)+timedelta(int(dict_values['IDAF'][0])-1)
        end_date=datetime(int(dict_values["IYR"][0])+int(dict_values['NBYR'][0])-1, 1, 1)+timedelta(int(dict_values['IDAL'][0])-1)
        simulation_days=(end_date-begin_date).days+1
        output_skip_years=int(dict_values["NYSKIP"][0])
        output_skip_days=(datetime(int(dict_values["IYR"][0])+output_skip_years, 1, 1)+timedelta(int(dict_values['IDAF'][0])-1)-begin_date).days
        begin_record=self.begin_date+timedelta(output_skip_days)
        
        self.model_infos["print_flag"]=int(dict_values["IPRINT"][0])
        self.model_infos["begin_date"]=begin_date
        self.model_infos["end_date"]=end_date
        self.model_infos["output_skip_years"]=output_skip_years
        self.model_infos["simulation_days"]=simulation_days
        self.model_infos["begin_record"]=begin_record
        
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
        
        self.model_infos["watershed_list"]=list(watershed.keys())
        self.model_infos["hru_list"] = list(itertools.chain.from_iterable(watershed.values()))
        self.model_infos["watershed_hru"]=watershed
        self.model_infos["n_hru"]=len(self.model_infos["hru_list"])
        self.model_infos["n_watershed"]=len(self.model_infos["watershed_list"])
        self.model_infos["n_rch"]=len(self.model_infos["watershed_list"])
        
        self.paras_file=pd.read_excel(os.path.join(self.work_path, 'SWAT_paras_files.xlsx'), index_col=0)
    
    def __del__(self):
        if self.used_temp_dir:
            os.makedirs(self.work_temp_dir)
    
    def _set_values(self, work_path, paras_values):
                
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
              
    def _get_observed_data(self):
        file_path=os.path.join(self.work_path, self.observed_file_name)
        rch_ids=[]
        rch_weights={}
        data=[]
        
        begin_record=self.model_infos["begin_record"]
        print_flag=self.model_infos["print_flag"]
        try:
            with open(file_path, "r") as f:
                lines=f.readlines()
                pattern_id=re.compile(r'REACH_ID_(\d+)\s+')
                pattern_value = re.compile(r'(\d+)\s+FLOW_OUT_(\d+)_(\d+)\s+(\d+\.?\d*)')
                
                total_reach=int(re.search(r'\d+', lines[0]).group()) #read the num of reaches
                obj_type=int(re.search(r'\d+', lines[1]).group()) #read the type of objective function  1-NSE 
                
                i=0;count_reach=0
                while i<len(lines):
                    line=lines[i]
                    match1= pattern_id.match(line)
                    if match1:
                        count_reach+=1
                        rch_id=int(match1.group(1))
                        rch_ids.append(rch_id)
                        rch_weight=float(re.search(r'\d+\.?\d*',lines[i+1]).group())
                        rch_weights[rch_id]=rch_weight
                        num_data=int(re.search(r'\d+', lines[i+2]).group())
                        i=i+3
                        
                        line=lines[i]
                        while pattern_value.match(line) is None:
                            i+=1
                            line=lines[i]   
                        n=0
                        while True:
                            line=lines[i];n+=1
                            match = pattern_value.match(line)
                            _, time, year = map(int, match.groups()[:-1])
                            value = float(match.groups()[-1])
                            if print_flag==0:
                                years=year-self.model_infos["begin_date"].year
                                if years==0:
                                    index=time-self.model_infos["begin_date"].month
                                else:
                                    index=time+12-self.model_infos["begin_date"].month+(years-1)*12
                            else:
                                index=(datetime(year, 1, 1)+timedelta(days=time-1)-self.model_infos["begin_record"]).days
                            data.append([rch_id, index, time, year,  value])
                            if n==num_data:
                                break
                            else:
                                i+=1              
                    i+=1
        except FileNotFoundError:
            raise FileNotFoundError("The observed data file is not found, please check the file name!")
        
        except Exception as e:
            raise ValueError("There is an error in observed data file, please check!")
        
        if total_reach!=count_reach:
            raise ValueError("The number of reaches in observed.txt is not equal to the number of reaches in flow data!")
                               
        observed_data = pd.DataFrame(data, columns=['rch_id', 'index', 'day', 'year', 'value'])
        rch_id_observed_data={}
        for rch_id in rch_ids:
            flow=observed_data.query('rch_id==@rch_id')          
            flow_value=flow['value'].to_numpy(dtype=float)
            data_index=flow['index'].to_numpy(dtype=int)
            ind_groups=self._get_lines_for_output(data_index)
            rch_id_observed_data[rch_id]=(ind_groups, flow_value)

        if sum(list(rch_weights.values()))!=1.0:
            raise ValueError("The sum of weights of observed data should be 1.0, please check observed.txt!")
        
        self.observe_infos["obj_type"]=obj_type
        self.observe_infos["rch_ids"]=rch_ids
        self.observe_infos["rch_flows"]=rch_id_observed_data
        self.observe_infos["rch_weights"]=rch_weights
    
    def _get_lines_for_output(self, index):
        
        index.ravel().sort()
        cur_group=[index[0]]; index_group=[]
        
        for i in range(1, len(index)):
            if index[i]==cur_group[-1]+1:
                cur_group.append(index[i])
            else:
                index_group.append([cur_group[0], cur_group[-1]])
                cur_group=[index[i]]
        
        index_group.append([cur_group[0], cur_group[-1]])
        return index_group
    
    def _generate_data_lines(self, group):
        start=group[0];end=group[-1]
        print_flag=self.model_infos["print_flag"]
        n_rch=self.model_infos["n_rch"]
        if print_flag==0:
            begin_month=self.model_infos["begin_record"].month
            first_period=13-begin_month
            
            # start=(start-out_skip_years)*n_rch+10
            pass
        elif print_flag==1:
            #
            pass
            
            
            
        
    
    def _record_default_values(self):
        """
        record default values from the swat file
        """
        var_infos_path=os.path.join(self.work_path, self.paras_file_name)
        low_bound=[]
        up_bound=[]
        disc_var=[]
        var_name=[]
        mode=[]
        assign_hru_id=[]
        discrete_bound=[]
        with open(var_infos_path, 'r') as f:
            lines=f.readlines()
            for line in lines:
                tmp_list=line.split()
                var_name.append(tmp_list[0])
                mode.append(tmp_list[1])
                op_type=tmp_list[2]
                lower_upper=tmp_list[3].split("_")
                
                if op_type=="c":
                    low_bound.append(float(lower_upper[0]))
                    up_bound.append(float(lower_upper[1]))
                    discrete_bound.append(0)
                    disc_var.append(0)
                else:
                    low_bound.append(float(lower_upper[0]))
                    up_bound.append(float(lower_upper[-1]))
                    discrete_bound.append([float(e) for e in lower_upper])
                    disc_var.append(1)
                    
                assign_hru_id.append(tmp_list[4:])
                
        self.lb= np.array(low_bound).reshape(1,-1)
        self.ub= np.array(up_bound).reshape(1,-1)
        self.mode= mode
        self.paras_list=var_name
        self.x_labels=self.paras_list
        self.disc_range=discrete_bound
        self.disc_var=disc_var
        self.n_input=len(self.paras_list)
        
        self.file_var_info={}
        
        self.data_types=[]
        watershed_hru=self.model_infos["watershed_hru"]
        watershed_list=self.model_infos["watershed_list"]
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
                if assign_hru_id[i][0]=="all":
                    files=[e+".{}".format(suffix) for e in self.hru_list]
                else:
                    files=[]
                    for ele in assign_hru_id[i]:
                        if "_" not in ele:
                            code=f"{'0' * (9 - 4 - len(ele))}{ele}{'0'*4}"
                            for e in watershed_hru[code]:
                                files.append(e+"."+suffix)
                        else:
                            bsn_id, hru_id=ele.split('_')
                            code=f"{'0' * (9 - 4 - len(bsn_id))}{bsn_id}{'0'*(4-len(hru_id))}{bsn_id}"
                            files.append(code+"."+suffix)
            elif suffix in self.watershed_suffix:
                if assign_hru_id[i][0]=="all":
                    files=[e+"."+suffix for e in watershed_list]
                else:
                    files=[e+"."+suffix for e in assign_hru_id[i]]
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
            
file_path="D:\YS_swat\TxtInOut"
temp_path="D:\\YS_swat\\instance_temp"
swat_exe_name="swat_681.exe"
observed_file_name="ob1.txt"
paras_file_name="paras_infos.txt"

swat_cup=SWAT_UQ_Flow(work_path=file_path,
                    paras_file_name=paras_file_name,
                    observed_file_name=observed_file_name,
                    swat_exe_name=swat_exe_name,
                    temp_path=temp_path,
                    max_threads=10, num_parallel=3)

from UQPyL.optimization import PSO
pso=PSO(problem=swat_cup)
res=pso.run()

