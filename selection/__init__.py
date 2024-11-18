from .base_selector import BaseSelector
from .k_center_selector import KCenterSelector

def get_selector(selector_name: str):
    if selector_name == "BaseSelector":
        return BaseSelector
    elif selector_name == "KCenterSelector":
        return KCenterSelector
    else:
        raise NotImplementedError(f"Selector {selector_name} not implemented")