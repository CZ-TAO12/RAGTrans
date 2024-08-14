import argparse

parser = argparse.ArgumentParser()

##### model parameters
parser.add_argument('-data_name', type=str, default='WEIXIN', choices=['WEIXIN', 'SMP', 'SIPD2020CHALLENGE'], help="dataset")
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-max_lenth', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=512)
parser.add_argument('--embSize', type=int, default=512, help='embedding size')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('-dropout', type=float, default=0.2)
#####data process
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
###save model
parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-patience', type=int, default=10, help="control the step of early-stopping")


