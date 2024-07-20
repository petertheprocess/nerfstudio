import json
import os
import numpy as np
from typing import Any, Dict, List, Tuple
from numpy.typing import NDArray
import logging
# set logging level to info to see the log
logging.basicConfig(level=logging.INFO)

class YCB_preprocess():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.data_list = []
        self.load_data()
        self.json_output = {}
    
    def render_intrinsic_to_json(self, K:NDArray[Any]):
        assert K.shape == (3, 3)
        self.json_output["fl_x"] = K[0, 0]
        self.json_output["fl_y"] = K[1, 1]
        self.json_output["cx"] = K[0, 2]
        self.json_output["cy"] = K[1, 2]
        # set k1 k2 k3 k4 p1 p2 to 0
        self.json_output["k1"] = 0
        self.json_output["k2"] = 0
        self.json_output["k3"] = 0
        self.json_output["k4"] = 0
        self.json_output["p1"] = 0
        self.json_output["p2"] = 0
        logging.info("Intrinsic matrix has been written to json_output")
        
