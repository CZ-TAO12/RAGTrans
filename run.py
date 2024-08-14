import datetime
import linecache
import pickle
import warnings
import numpy as np
import torch
import scipy
import Constants
from tqdm import tqdm
from models.mmmodels import *
from torch_geometric.loader import DataLoader
from dataLoader import datasets, Read_data, Split_data
from utils.parsers import parser
from utils.EarlyStopping import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import logging


metric = Metrics()
opt = parser.parse_args() 

def init_seeds(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def model_training(model, train_loader, epoch):
    ''' model training '''
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0

    print('start training: ', datetime.datetime.now())
    # training
    model.train()
    with tqdm(total=len(train_loader)) as t:
        for step, batch in enumerate(train_loader):
            
            hg_idx = trans_to_cuda(batch.edge_index)
            pid = trans_to_cuda(batch.pid)
            popularity = batch.y
            uid = trans_to_cuda(batch.uid)
            related_items = trans_to_cuda(batch.related_items)

            model.zero_grad()
        
            tar = trans_to_cuda(popularity)
            
            pred = model(pid, hg_idx, related_items, popularity, uid)
            pred = pred.view_as(tar)
       
            loss = model.loss_function(pred, tar)

            loss.backward()
            model.optimizer.step()

            ### tqdm parameter
            t.set_description(desc="Epoch %i" % epoch)
            t.set_postfix(steps=step, loss=loss.data.item())
            t.update(1)

            total_loss += loss.item()

        print('\tTotal Loss:\t%.3f' % total_loss)

        return total_loss

def model_testing(model, test_loader):
    ''' Epoch operation in evaluation phase '''

    total_mse = 0.0
    total_mae = 0.0
    total_src = 0.0

    n_total_words = 0.0

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    batch_num = 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):

            hg_idx = batch.edge_index
            pid = batch.pid
            uid = trans_to_cuda(batch.uid)
            popularity = batch.y

            bs = popularity.size()[0]
            n_total_words += bs

            related_items = trans_to_cuda(batch.related_items)

            hg_idx = trans_to_cuda(torch.LongTensor(np.array(hg_idx)))
       
            pid_int = list(int(x) for x in pid)

            pid = trans_to_cuda(torch.from_numpy(np.array(pid_int)))
            tar = trans_to_cuda(popularity.float()).view(-1).detach().cpu()

            pred = model(pid, hg_idx, related_items, popularity, uid)
            pred = np.array(pred.view_as(tar).detach().cpu())
            tar = np.array(tar)
            mse = np.mean((pred - tar) ** 2)
            mae = np.mean(np.abs(pred - tar))
            corr, pvalue = scipy.stats.spearmanr(tar, pred)
            total_mse += mse* bs
            total_mae += mae* bs
            total_src += corr* bs
            print(f'tar:{tar} pred:{pred}')
        return total_mse/n_total_words, total_mae/n_total_words, total_src/n_total_words
      
def train_test(epoch, model, train_loader, val_loader, test_loader):

    total_loss = model_training(model, train_loader, epoch)
    val_scores = model_testing(model, val_loader)
    test_scores = model_testing(model, test_loader)

    return total_loss, val_scores, test_scores

def main(data_path, seed=2023):

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(opt.data_name+'.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    init_seeds(seed)

    train, valid, test = Read_data(data_path)  ##

    train_data = datasets(train)
    val_data = datasets(valid)
    test_data = datasets(test)

    #### Build DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)


    # ========= Early_stopping =========#
    save_model_path = opt.save_path + 'MMHG.pt'
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=save_model_path)

    # ========= Building Model =========#
    model = trans_to_cuda(MMHG(args=opt, dropout=opt.dropout))


    validation_history = np.Inf

    test_mse = 0.0
    test_mae = 0.0
    test_src = 0.0

    for epoch in range(opt.epoch):
        total_loss, val_scores, test_scores = train_test(epoch, model, train_loader, val_loader, test_loader)
        print(f'train loss:{total_loss}')
        print(f'validation mse:{val_scores[0]}')
        print(f'validation mae:{val_scores[1]}')
        print(f'validation src:{val_scores[2]}')
        print(f'test mse:{test_scores[0]}')
        print(f'test mae:{test_scores[1]}')
        print(f'test src:{test_scores[2]}')

        if validation_history >= val_scores[0]:
            validation_history =val_scores[0]

            test_mse = test_scores[0]
            test_mae = test_scores[1]
            test_src = test_scores[2]
        
        early_stopping(val_scores[0], model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break
    
    # ========= Final score =========#
    print(" -(Finished!!) \n test scores: ")
    print("--------------------------------------------")
    print(f'test mse:{test_mse}')
    print(f'test mae:{test_mae}')
    print(f'test src:{test_src}')
    logger.info(f'test mse:{test_mse}, test mae:{test_mae}, test src:{test_src}')

if __name__ == "__main__":
    print(opt)
    main(opt.data_name, seed=2023)

