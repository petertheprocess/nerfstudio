import json
import numpy as np

class DnerfJsonConverter:
    def __init__(self, JsonPath):
        self.json_metadata = json.load(open(JsonPath))
        self.jsonPath = JsonPath
    
    def add_camera_angle_x(self, JsonPath):
        fx = self.json_metadata['fl_x']
        w = self.json_metadata['w']
        fov_x = 2 * np.arctan(w / (2 * fx))
        self.json_metadata['camera_angle_x'] = fov_x
        

    def add_time_dnerf(self, time_total_sec=1.0):
        frames = self.json_metadata['frames']
        len_frames = len(frames)
        time_step = time_total_sec / (len_frames-1)
        for index, frame in enumerate(frames):
            frame['time'] = time_step * index
    
    def save_json(self):
        with open(self.jsonPath.replace('.json','_dynamic.json'), 'w') as f:
            json.dump(self.json_metadata, f, indent=4)
        
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--JsonPath", type=str, required=True)
    args = parser.parse_args()
    converter = DnerfJsonConverter(args.JsonPath)
    converter.add_camera_angle_x(args.JsonPath)
    converter.add_time_dnerf()
    converter.save_json()

