from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset

from .tensorf_init import TensoRF_Init_Dataset
from .shapeBufferLoader import ShapeBufferDataset
from .tensoIR_rotation_setting import TensoIR_Dataset_unknown_rotated_lights
from .tensoIR_relighting_test import tensoIR_Relighting_test
from .tensoIR_simple import TensoIR_Dataset_simple
from .tensoIR_lightstage import TensoIR_Dataset_lightstage
from .tensoIR_lightstage_multi_light import TensoIR_Dataset_lightstage_multi_light
from .tensoIR_material_editing_test import tensoIR_Material_Editing_test
from .tensoIR_general_multi_lights import TensoIR_Dataset_unknown_general_multi_lights

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'tankstemple': TanksTempleDataset,
                'nsvf': NSVF,
                'own_data': YourOwnDataset,
                'tensorf_init': TensoRF_Init_Dataset,
                'shapeBuffer': ShapeBufferDataset,
                'tensoIR_unknown_rotated_lights':TensoIR_Dataset_unknown_rotated_lights,
                'tensoIR_unknown_general_multi_lights': TensoIR_Dataset_unknown_general_multi_lights,
                'tensoIR_relighting_test':tensoIR_Relighting_test,
                'tensoIR_material_editing_test':tensoIR_Material_Editing_test,
                'tensoIR_simple':TensoIR_Dataset_simple,
                'tensoIR_lightstage':TensoIR_Dataset_lightstage,
                'tensoIR_lightstage_multi_light': TensoIR_Dataset_lightstage_multi_light
                }
