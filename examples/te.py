import re
import os
import itertools

def _set_name_vlue(file_path, name, value):
    value=str(value)
    pattern=r"(\s*)(\d+\.?\d*)(\s*.*{}.*)".format(name)
    with open(file_path, "r+") as f:
        lines=f.readlines()
        for i, line in enumerate(lines):
            pattern
            match = re.search(pattern, line)
            
            
            if match:
                pattern_value=match.group(2)
                
                if re.fullmatch(r'\d+', pattern_value) is not None and re.fullmatch(r'\d+', value) is not None: 
                    new_text = re.sub(pattern,  match.group(1)+"{}".format(value)+match.group(3), line)
                    lines[i]=new_text
                else:
                    raise ValueError("Please check the type of input value for {} in {}".format(name, file_path))
                
                break
        f.seek(0)
        f.writelines(lines)
        f.truncate()

work_path="H:\SiHuRiver\model\FuTIanSi001\Scenarios\Test\TxtInOut"
HRU_suffix=[".chm", ".gw", ".hru", ".mgt", ".sdr", ".sep", ".sol"]
Watershed_suffiex=[".pnd", ".rte", ".sub", ".swq", "wgn", "wus"]

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


for sub in total_sub_list:
    path=os.path.join(work_path, sub+".mgt")
    _set_name_vlue(path, "URBLU", 2)

text="     0    | IRRSC: irrigation code"
