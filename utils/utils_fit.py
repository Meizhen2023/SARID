# -*- coding: utf-8 -*-
# @Time : 2023/1/9 20:01 
# @Author : Mingzheng 
# @File : utils_fit.py
# @desc :
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F

from utils.utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, fp16, scaler, backbone, save_period, save_dir, local_rank=0):

    total_loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_features, batch_rainfall_intensities = batch
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            if backbone in [ "resnet50","Transformer","EcapaTdnn","CNN","DRNN","GCNN"]:
                rainfall_intensities = model_train.cuda()(batch_features)
                loss = F.mse_loss(rainfall_intensities,batch_rainfall_intensities.to(torch.float))
                # print(rainfall_intensities[0:10])
                # print(batch_rainfall_intensities[0:10])
                total_loss += loss.item()
            else:
                rainfall_intensities = model_train(batch_features)
                loss = 0
                index = 0
                for rainfall_intensity in rainfall_intensities:
                    loss = F.mse_loss(rainfall_intensity,batch_rainfall_intensities.view([-1, 1]))
                    index += 1
                total_loss += loss.item() / index

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                if backbone [ "resnet50","Transformer","EcapaTdnn","CNN","DRNN","GCNN"]:
                    rainfall_intensities = model_train(batch_features)
                    loss = F.mse_loss(rainfall_intensities,batch_rainfall_intensities)
                    total_loss += loss.item()
                else:
                    rainfall_intensities = model_train(batch_features)
                    loss = 0
                    index = 0
                    for rainfall_intensity in rainfall_intensities:
                        loss = F.mse_loss(rainfall_intensity, batch_rainfall_intensities)
                        index += 1
                    total_loss += loss.item() / index


            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_features, batch_rainfall_intensities = batch

            if backbone in [ "resnet50","Transformer","EcapaTdnn","CNN","DRNN","GCNN"]:
                rainfall_intensities = model_train(batch_features)
                loss = F.mse_loss(rainfall_intensities, batch_rainfall_intensities)
                val_loss += loss.item()
            else:
                rainfall_intensities = model_train(batch_features)
                loss = 0
                index = 0
                for rainfall_intensity in rainfall_intensities:
                    loss = F.mse_loss(rainfall_intensity, batch_rainfall_intensities)
                    index += 1
                val_loss += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f-regression.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights_regression.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights_regression.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights_regression.pth"))

