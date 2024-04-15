import argparse
import os
from Utils.zeroshot import zeroshot_inference


def parse_args():
    """ parsing the arguments that are used in for zero-shot inference using UrbanCLIP"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='primary',
                        choices=['primary', 'multi', 'transfer-london', 'transfer-singapore',],
                        help='can be primary, multi, transfer-london, or transfer-singapore')
    parser.add_argument('--taxonomy', type=str, default='UrbanCLIP', choices=['UrbanCLIP', 'function_name'],
                        help='can be UrbanCLIP or function_name (only using function name prompts)')
    parser.add_argument('--prompt_template', type=str, default='UrbanCLIP',
                        choices=['UrbanCLIP', 'Wu', 'Photo', 'CLIP80', 'no_template', 'UrbanCLIP_SC', 'Wu_without_SC'],
                        help= 'can be UrbanCLIP, Wu, Photo, CLIP80, no_template, UrbanCLIP_SC, Wu_without_SC, and please refer to the paper for more details'
                        )
    parser.add_argument('--ensemble', type=str, default='zpe', choices=['mean', 'zpe'])
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    args = parse_args()
    zeroshot_inference(args)

