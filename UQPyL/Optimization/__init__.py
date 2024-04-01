from .ga import GA
from .boxmin import Boxmin
from .adam import Adam
from .sce_ua import SCE_UA
from .asmo import ASMO
from .nsga_ii import NSGAII
from .mo_asmo import MOASMO
__all__=[
    'GA',
    'Boxmin',
    'Adam',
    'SCE_UA',
    'ASMO',
    'NSGAII',
    'MOASMO',
         ]

MP_List=['Boxmin']
EA_List=['GA']