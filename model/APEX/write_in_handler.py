from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings

@dataclass
class SubEntry:
    offset: int
    original_val: float

@dataclass
class Parameter:
    name: str
    index: int
    entry: SubEntry               
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
        self,
        name: str,
        index: int,
        mode: int,
        typ: int,
        linePos: int,   # 1-based
        staPos: int,    # 1-based Start
        width: int,
        precision: int,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ) -> bool:
        
        line_idx = linePos - 1
        
        if typ == 1:
            if lb is not None:
                lb = int(lb)
            if ub is not None:
                ub = int(ub)

        line_start = self.line_offsets[line_idx]

        off = line_start + (staPos - 1)

        field = bytes(self.base_content[off:off + width])
        val = self._parse_float_field(field)

        entry = SubEntry(offset=off, original_val=val)

        self.params[index] = Parameter(
            name = name,
            index = index,
            entry = entry,
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
        warn_stacklevel: int = 2,
        warn_detail_limit: int = 20,
    ):
        
        self.file_content = bytearray(self.base_content)

        mods: List[Modification] = []
        clamp_events = []

        for idx, input_val in zip(indices, vals):
            
            p = self.params.get(idx)
            if not p:
                continue

            e = p.entry

            # 1) calculate raw
            if p.mode == 0:
                raw = e.original_val * (1.0 + float(input_val))
            elif p.mode == 1:
                raw = float(input_val)
            elif p.mode == 2:
                raw = e.original_val + float(input_val)
            else:
                raw = float(input_val)

            # 2) transform type
            raw2 = int(raw) if p.typ == 1 else raw

            # 3) clamp (warning)
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
            self.file_content[m.offset:m.offset + m.width] = m.data

        if clamp_events:
            head = clamp_events[:warn_detail_limit]
            msg_lines = [
                f"Param clamp: idx={i}, name={name}, {raw} -> {clamped}, hardBounds=[{lb},{ub}]"
                for (i, name, raw, clamped, lb, ub) in head
            ]
            more = "" if len(clamp_events) <= warn_detail_limit else (
                f"\n... and {len(clamp_events) - warn_detail_limit} more clamps"
            )
            warnings.warn("\n".join(msg_lines) + more, stacklevel=warn_stacklevel)

        with open(output_filepath, "wb") as out:
            out.write(self.file_content)