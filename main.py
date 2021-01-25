import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset 
import yaml
import IPython
import pdb
import tqdm
import argparse

from torch import nn
from tqdm import trange
from model import Model
from dataset import Resisitivity_Dataset
import wandb

def train(expert, optimizer, criterion, dataloader, eval_dataloader, config):
    

    step = 0
    # IPython.embed()
    # pdb.set_trace()
    for epoch in trange(config['train_args']['epoches']):
        for num, (data, label) in enumerate(tqdm.tqdm(dataloader)):
            # forward model to get prediction
            output = expert(data).squeeze(-1)
            
            # calculate loss 
            loss = criterion(output, label)
            wandb.log({"loss":loss.item()},step=step)        
            # compute gradient and accumulate gradient in each weights
            loss.backward()
            
            # update all weights in model
            optimizer.step()
            
            # clean gradient on each weights for preparing next update step
            optimizer.zero_grad()
            
            step +=1
            
            if step % config['train_args']['eval_step'] == 0:
                # evaluate the prediction whether is close to the ground truth
                eval_loss = evaluate(eval_dataloader, expert, config['eval_part'],step)
                torch.save(expert.state_dict(), f"step-{step}_"+config['path']['model_save_path'])
    
    return 

def evaluate(eval_dataloader, model, config_dict,step):
    
    criterion = nn.L1Loss()
    prediction_list = []
    label_list = []
    eval_loss = 0
    with torch.no_grad():
        for num, (data, label) in enumerate(tqdm.tqdm(eval_dataloader)):
            prediction = model(data).squeeze(-1)
            loss=criterion(prediction, label)
            eval_loss += loss

            # save all prediction to visualize result
            prediction = prediction.tolist()
            prediction_list.extend(prediction)
            label_list.extend(label.tolist())
    eval_loss /= (num+1)
    wandb.log({"eval_loss":eval_loss.item()},step=step)
    
    # write the (prediction, label) to the output file
    zip_object = zip(prediction_list, label_list)
    zipped_list = list(zip_object)

    f = open(f"step-{step}_" + config_dict['output_file_path'], "w")
    for x in zipped_list:
        f.write(str(x)+"\n")
    f.close()

def main(config):
    
    # wandb init 
    wandb.init(project=config['project_name'], config=config,name=config['exp_name'])
    
    # Initialize Model
    expert = Model(config['model'])
    wandb.watch(expert)

    # Define optimizer
    optimizer = torch.optim.Adam(expert.parameters(),lr=config['train_args']['lr'])

    # Initialize dataset
    train_dataset = Resisitivity_Dataset(config['path']['data_path'], config, "train")
    eval_dataset = Resisitivity_Dataset(config['path']['data_path'], config, "eval")

    # Build dataloader
    train_dataloader = DataLoader(train_dataset,batch_size=config['train_args']['batch_size'], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset,batch_size=config['train_args']['batch_size'])

    # Define loss function
    criterion = torch.nn.L1Loss()

    # training
    train(expert, optimizer, criterion, train_dataloader, eval_dataloader, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="config_default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config,"r"))
    
    main(config)