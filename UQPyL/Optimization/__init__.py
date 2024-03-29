from .GA import GA
from .boxmin import Boxmin
from .adam import Adam
from .sce_ua import SCE_UA
from .asmo import ASMO
from .nsga_ii import NSGAII
__all__=[
    'GA',
    'Boxmin',
    'Adam',
    'SCE_UA',
    'ASMO',
    'NSGAII'
         ]

MP_List=['Boxmin']
EA_List=['GA']