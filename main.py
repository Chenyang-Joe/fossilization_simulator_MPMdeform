
import argparse
import json
from libs.simulator_deform import *

def read_json_from_arg():
    parser = argparse.ArgumentParser(description="Fossilization Simulator MPMdeform")
    parser.add_argument('--json', type=str, required=True, help='Path to the input JSON file')

    args = parser.parse_args()

    with open(args.json, 'r') as f:
        data = json.load(f)

    return data

if __name__ == "__main__":
    print("hello")
    json_data = read_json_from_arg()
    myDeformSim = SimulatorDeform(json_data)
    myDeformSim.preprocess()
    myDeformSim.generate_PC()
    myDeformSim.MPMPytorch_deform()