
import argparse
import json
from libs.simulator_deform import *
from libs.simulator_deform_fragments import *
import numpy as np
import time

def read_json_from_arg():
    parser = argparse.ArgumentParser(description="Fossilization Simulator MPMdeform")
    parser.add_argument('--json', type=str, required=True, help='Path to the input JSON file')
    # make --fragment_model optional with default False
    parser.add_argument('--fragment_model', action='store_true', help='Whether the input model is fragmented')

    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)

    return data, args.fragment_model

if __name__ == "__main__":
    print("hello")
    np.random.seed(int(time.time()))
    json_data, fragment_model = read_json_from_arg()
    if not fragment_model:
        myDeformSim = SimulatorDeform(json_data)
        myDeformSim.preprocess()
        myDeformSim.generate_PC()
        myDeformSim.MPMPytorch_deform()
        myDeformSim.Mesh_reconstruction()
        myDeformSim.clean_up()
        print("Successfully finish the deformation simulation!")
    else:
        myDeformSimFrag = SimulatorDeformFragments(json_data)
        myDeformSimFrag.preprocess()
        myDeformSimFrag.generate_PC()
        myDeformSimFrag.MPMPytorch_deform()
        myDeformSimFrag.Mesh_reconstruction()
        myDeformSimFrag.clean_up()
        print("Successfully finish the deformation simulation with fragments!")
