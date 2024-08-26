import argparse

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SimHGCL")

    parser.add_argument("--dataset", default="amazon", help="Dataset to use")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning Rate")
    parser.add_argument('--ssl_lambda', type=float, default=0.02, help='cl_rate_2')#0.02
    parser.add_argument("--reg_lambda", type=float, default=0.0001, help="Regularizations")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature")
    parser.add_argument("--sparsity_test", type=int, default=0, help="sparsity_test")
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--top_K', type=str, default="[20, 10, 5]", help='size of Top-K')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=300, help='batch size')
    parser.add_argument("--verbose", type=int, default=1, help="Test interval")
    parser.add_argument('--GCN_layer', type=int, default=2, help="the layer number of GCN")
    parser.add_argument('--col_D_inv', type=int, default=-0.3, help="col_D_inv")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument("--device", type=str, default="cuda:3", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    
    return parser.parse_args()
