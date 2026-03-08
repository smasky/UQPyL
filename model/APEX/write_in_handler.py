from dataclasses import dataclass
from typing import Dict, List, Optional
import os

_WARNING_PROMPTED = False

@dataclass
class SubEntry:
    offset: int
    original_val: float

@dataclass
class Parameter:
    name: str
    index: int
    entries: List[SubEntry]               
    mode: int                     # 0: orig*(1+x), 1: =x, 2: orig+x
    typ: int                      # 1: int, else: float
    precision: int
    width: int
    lb: Optional[float] = None
    ub: Optional[float] = None

@dataclass
class Modification:
    offset: int
    width: int
    data: bytes

class WriteInHandler:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, "rb") as f:
            content = bytearray(f.read())

        self.base_content = content
        self.file_content = bytearray(content)

        self.line_offsets = self._build_line_index()
        self.params: Dict[int, Parameter] = {}

    def _build_line_index(self) -> List[int]:
        offs = [0]
        pos = 0
        n = len(self.base_content)
        data = self.base_content
        while True:
            j = data.find(b"\n", pos)
            if j == -1:
                break
            if j + 1 < n:
                offs.append(j + 1)
            pos = j + 1
        return offs

    @staticmethod
    def _format_value(val: float, width: int, precision: int, typ: int) -> bytes:
        if typ == 1:
            s = str(int(val))
        else:
            s = f"{float(val):.{precision}f}"
        if len(s) > width:
            return b"*" * width
        return s.rjust(width).encode("ascii")

    @staticmethod
    def _parse_float_field(b: bytes) -> Optional[float]:
        s = b.decode("ascii", errors="ignore").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def register_param(
        self, spec, lib_info, hardBound: bool = True
    ) -> bool:
        
        name = spec.name
        index = spec.index
        mode = spec.mode
        typ = lib_info.type
        staPos = lib_info.file.start
        width = lib_info.file.width
        precision = lib_info.file.precision
        lb = lib_info.lb if hardBound else None
        ub = lib_info.ub if hardBound else None
        maxNum = lib_info.file.maxNum
        
        if typ == 1:
            if lb is not None:
                lb = int(lb)
            if ub is not None:
                ub = int(ub)

        linePos_data = lib_info.file.line
        lines = linePos_data if isinstance(linePos_data, list) else [linePos_data]
        
        entries_list = []
        
        for linePos in lines:
            line_idx = linePos - 1

            if line_idx >= len(self.line_offsets):
                continue

            line_start = self.line_offsets[line_idx]
            
            if line_idx + 1 < len(self.line_offsets):
                line_end = self.line_offsets[line_idx + 1]
            else:
                line_end = len(self.base_content)
                
            base_offset = line_start + (staPos - 1)
            
            row_entries = self._scan_entries(
                start_offset=base_offset, 
                width=width, 
                max_num=maxNum, 
                line_end_offset=line_end
            )
           
            entries_list.extend(row_entries)

        self.params[index] = Parameter(
            name = name,
            index = index,
            entries = entries_list,
            mode = mode,
            typ = typ,
            precision = precision,
            width = width,
            lb = lb,
            ub = ub
        )
        
        return True

    def set_values_and_save(
            self,
            output_filepath: str,
            indices: List[int],
            vals: List[float],
            backup_path: Optional[str] = None,
            warn_detail_limit: int = 20,
        ):
            
            self.file_content = bytearray(self.base_content)
            mods: List[Modification] = []
            clamp_events = []

            for idx, input_val in zip(indices, vals):
                p = self.params.get(idx)
                if not p:
                    continue

                for e in p.entries:
                    
                    if p.mode == 0:   # relative %
                        raw = e.original_val * (1.0 + float(input_val))
                    elif p.mode == 1: # replace
                        raw = float(input_val)
                    elif p.mode == 2: # absolute add
                        raw = e.original_val + float(input_val)
                    else:
                        raw = float(input_val)

                    raw2 = int(raw) if p.typ == 1 else raw

                    clamped = raw2
                    if p.lb is not None and clamped < p.lb:
                        clamped = p.lb
                    if p.ub is not None and clamped > p.ub:
                        clamped = p.ub

                    if clamped != raw2:
                        clamp_events.append((idx, p.name, raw2, clamped, p.lb, p.ub))

                    b = self._format_value(clamped, p.width, p.precision, p.typ)
                    mods.append(Modification(offset=e.offset, width=p.width, data=b))

            for m in mods:
                self.file_content[m.offset : m.offset + m.width] = m.data

            if clamp_events:
                head = clamp_events[:warn_detail_limit]
                msg_lines = [
                    f"Param clamp: {output_filepath}, idx={i}, raw={raw:.2f}->{clamped:.2f}"
                    for (i, name, raw, clamped, lb, ub) in head
                ]
                more = "" if len(clamp_events) <= warn_detail_limit else f"\n... {len(clamp_events)-warn_detail_limit} more"
                full_msg = "\n".join(msg_lines) + more
                if backup_path:
                    wpath = os.path.join(backup_path, "warning.txt")
                    with open(wpath, "a", encoding="utf-8") as wf:
                        wf.write(full_msg + "\n")
                    global _WARNING_PROMPTED
                    if not _WARNING_PROMPTED:
                        print(f"Warnings saved to {wpath}, please check.")
                        _WARNING_PROMPTED = True

            with open(output_filepath, "wb") as out:
                out.write(self.file_content)
            
    def _scan_entries(
        self, start_offset: int, width: int, max_num: int, line_end_offset: int
    ) -> List[SubEntry]:

        entries = []
        
        for i in range(max_num):
            curr_off = start_offset + (i * width)
            curr_end = curr_off + width

            if curr_end > line_end_offset:
                break
            
            if curr_end > len(self.base_content):
                break

            field = bytes(self.base_content[curr_off : curr_end])
            val = self._parse_float_field(field)

            if val is None:
                break 

            entries.append(SubEntry(offset=curr_off, original_val=val))
            
        return entries