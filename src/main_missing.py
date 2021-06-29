# new version, 2020/08/19

import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil
import gc

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True


_, config = load_config_yaml('config.yaml')
config['is_discrim_s'] = True if config['lambda_adv_s'] > 0 else False
config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)


# ckpt folder, load yaml config
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase' or key == 'continue_train':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
        config['is_discrim_s'] = True if config['lambda_adv_s'] > 0 else False
        config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
    else:
        save_config_yaml(config['ckpt_path'], config)

print(config['model_name'])
# config['ckpt_name'] = 'model_best.pth.tar'
# pdb.set_trace()

Data = ZeroDoseDataAll(config['dataset_name'], config['data_path'], norm_type=config['norm_type'], batch_size=config['batch_size'], num_fold=config['num_fold'], \
                    fold=config['fold'], shuffle=config['shuffle'], num_workers=0, block_size=config['block_size'], \
                    contrast_list=config['contrast_list'], aug=False, dropoff=config['dropoff'], skull_strip=config['skull_strip'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
# valDataLoader = Data.testLoader
testDataLoader = Data.testLoader

# define model
if config['model_name'] == 'MultimodalModel':
    # model = MultimodalModel(input_size=(config['input_height'], config['input_width']), \
    #                         modality_num=len(config['contrast_list']), in_num_ch=2*config['block_size']+1, \
    #                         out_num_ch=1, s_num_ch=8, device=config['device'])#.to(config['device'])  #old
    if config['dataset_name'] == 'BraTS' or config['norm_type'] == 'z-score':
        config['target_output_act'] = 'no'
        print('output activation: no-------------------')
    else:
        config['target_output_act'] = 'softplus'



    if config['norm_type'] == 'mean':
        config['input_output_act'] = 'softplus'
    else:
        config['input_output_act'] = 'no'
    model = MultimodalModel(input_size=(config['input_height'], config['input_width']), \
                            modality_num=len(config['contrast_list']), in_num_ch=2*config['block_size']+1, \
                            out_num_ch=config['out_num_ch'], s_num_ch=config['s_num_ch'], z_size=config['z_size'], is_cond=config['is_cond'], \
                            is_discrim_s=config['is_discrim_s'], is_distri_z=config['is_distri_z'], \
                            s_compact_method=config['s_compact_method'], s_sim_method=config['s_sim_method'], z_sim_method=config['z_sim_method'], \
                            shared_ana_enc=config['shared_ana_enc'], shared_mod_enc=config['shared_mod_enc'], \
                            shared_inp_dec=config['shared_inp_dec'], device=config['device'],\
                            input_output_act=config['input_output_act'], target_output_act=config['target_output_act'], \
                            target_model_name=config['target_model_name'], fuse_method=config['fuse_method'], others=config['others'])


    # pdb.set_trace()
else:
    raise ValueError('not supporting other models yet!')

# define optimizer
# pdb.set_trace()
if config['fix_pretrain'] and config['continue_train']:
    print('----------------Fixed stage 1 parts!------------------')
    for submodel in model.anatomy_encoder_enc_list:
        for param in submodel.parameters():
            param.requires_grad = False
    for param in model.anatomy_encoder_dec.parameters():
        param.requires_grad = False
    for submodel in model.modality_encoder_list:
        for param in submodel.parameters():
            param.requires_grad = False
    for submodel in model.input_decoder_list:
        for param in submodel.parameters():
            param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

if config['is_discrim_s']:
    optimizer_d_s = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
    if config['is_discrim_s']:
        try:
            [optimizer_d_s], _ =  load_checkpoint_by_key([optimizer_d_s], config['ckpt_path'], ['optimizer_d_s'], config['device'], config['ckpt_name'])
        except:
            print('Pretrained model does not have discriminator')
else:
    start_epoch = -1

if config['phase'] == 'train':
    save_config_file(config)

# train
def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    process = psutil.Process(os.getpid())

    # stat = evaluate(phase='val', set='val', save_res=False)
    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'recon_y': 0., 'recon_y_fused': 0., 'recon_x': 0., 'recon_x_mix': 0., 'kl': 0.,
                        'latent_z': 0., 'sim_s': 0., 'sim_z': 0., 'adv_s': 0., 'adv_s_d': 0., 'all': 0.}
        global_iter0 = global_iter
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            # print(iter)
            # print(process.memory_info().rss)
            # print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            # gc.collect()
            # continue


            inputs = sample['inputs'].to(config['device'], dtype=torch.float)
            inputs_list = []
            for idx in range(len(config['contrast_list'])):
                inputs_list.append(inputs[:, idx*(2*config['block_size']+1):(idx+1)*(2*config['block_size']+1)])

            targets = sample['targets'].to(config['device'], dtype=torch.float)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            mask_img = sample['mask_img'].to(config['device'], dtype=torch.float)

            # pred, _ = model(inputs_list)
            si_list = model.compute_anatomy_encoding(inputs_list, mask_img)
            zi_list, zi_mean_list, zi_log_var_list = model.compute_modality_encoding(inputs_list, si_list, phase='train')
            xi_fake_list = model.reconstruct_input_si_zi(si_list, zi_list)
            xi_fake_mix_list = model.reconstruct_input_si_zj(si_list, zi_list)


            # recon y
            if iter == 0:
                # to simplify, only compute y once, if no loss on recon y
                y_fake_list = model.reconstruct_output_si(si_list)
                y_fake_fused = model.reconstruct_output_si_fused(si_list, mask)
            else:
                if config['lambda_recon_y'] > 0:
                    y_fake_list = model.reconstruct_output_si(si_list)
                if config['lambda_recon_y_fused'] > 0:
                    y_fake_fused = model.reconstruct_output_si_fused(si_list, mask)

            loss = 0
            if config['lambda_recon_y'] > 0:
                if config['dataset_name'] == 'BraTS':
                    loss_recon_y = model.compute_segmentation_loss_y_list(targets, y_fake_list, mask)
                else:
                    loss_recon_y = model.compute_recon_loss_y_list(targets, y_fake_list, mask, p=config['p'])
                loss += config['lambda_recon_y'] * loss_recon_y
            else:
                loss_recon_y = torch.tensor(0.)
            if config['lambda_recon_y_fused'] > 0:
                if config['dataset_name'] == 'BraTS':
                    loss_recon_y_fused = model.compute_segmentation_loss_y(targets, y_fake_fused)
                else:
                    loss_recon_y_fused = model.compute_recon_loss_y(targets, y_fake_fused, p=config['p'])
                loss += config['lambda_recon_y_fused'] * loss_recon_y_fused
            else:
                loss_recon_y_fused = torch.tensor(0.)
            if config['lambda_recon_x'] > 0:
                loss_recon_x = model.compute_recon_loss_x_list(inputs_list, xi_fake_list, mask, p=config['p'])
                loss += config['lambda_recon_x']*loss_recon_x
            else:
                loss_recon_x = torch.tensor(0.)
            if config['lambda_recon_x_mix'] > 0:
                loss_recon_x_mix = model.compute_recon_loss_x_mix_list(inputs_list, xi_fake_mix_list, mask, p=config['p'])
                loss += config['lambda_recon_x_mix']*loss_recon_x_mix
            else:
                loss_recon_x_mix = torch.tensor(0.)
            if config['lambda_kl'] > 0:
                if config['is_distri_z']:
                    zi_prior_mean_list, zi_prior_log_var_list = model.compute_zi_prior_distribution(targets.shape[0], len(config['contrast_list']), config['device'])
                    loss_kl = model.compute_kl_loss_list_two_gaussian(zi_mean_list, zi_log_var_list, zi_prior_mean_list, zi_prior_log_var_list, mask)
                else:
                    loss_kl = model.compute_kl_loss_list_standard(zi_mean_list, zi_log_var_list, mask)
                loss += config['lambda_kl']*loss_kl
            else:
                loss_kl = torch.tensor(0.)
            if config['lambda_latent_z'] > 0:
                # latent back x_fake
                si_list_new = model.compute_anatomy_encoding(xi_fake_list, mask_img)
                zi_list_new, zi_mean_list_new, zi_log_var_list_new = model.compute_modality_encoding(xi_fake_list, si_list_new, phase='train')
                loss_latent_z = model.compute_latent_z_loss(zi_mean_list, zi_mean_list_new, mask)
                loss += config['lambda_latent_z']*loss_latent_z
            else:
                loss_latent_z = torch.tensor(0.)
            if config['lambda_sim_s'] > 0:
                loss_sim_s = model.compute_similarity_s_loss(si_list, mask)
                loss += config['lambda_sim_s']*loss_sim_s
            else:
                loss_sim_s = torch.tensor(0.)
            if config['lambda_sim_z'] > 0:
                loss_sim_z = model.compute_similarity_z_loss(zi_list, mask)
                loss += config['lambda_sim_z']*loss_sim_z
            else:
                loss_sim_z = torch.tensor(0.)
            if config['lambda_adv_s'] > 0:
                loss_adv_s_d, loss_adv_s = model.compute_adversarial_loss(si_list, mask)
                loss += config['lambda_adv_s']*loss_adv_s
            else:
                loss_adv_s = torch.tensor(0.)
                loss_adv_s_d = torch.tensor(0.)

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['recon_y_fused'] += loss_recon_y_fused.item()
            loss_all_dict['recon_x'] += loss_recon_x.item()
            loss_all_dict['recon_x_mix'] += loss_recon_x_mix.item()
            loss_all_dict['kl'] += loss_kl.item()
            loss_all_dict['latent_z'] += loss_latent_z.item()
            loss_all_dict['sim_s'] += loss_sim_s.item()
            loss_all_dict['sim_z'] += loss_sim_z.item()
            loss_all_dict['adv_s'] += loss_adv_s.item()
            loss_all_dict['adv_s_d'] += loss_adv_s_d.item()
            loss_all_dict['all'] += loss.item()

            if torch.isnan(loss):
                pdb.set_trace()

            if config['lambda_adv_s'] > 0:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            # print(name, torch.isfinite(param.grad).all())
            # accumulate gradient
            if (iter+1) % (16 // config['batch_size']) == 0:
                optimizer.step()
                optimizer.zero_grad()

                if config['lambda_adv_s'] > 0:
                    optimizer_d_s.zero_grad()
                    loss_adv_s_d.backward()
                    optimizer_d_s.step()

            if global_iter % 10 == 0:
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon x=[%.4f], recon x_mix=[%.4f], recon y=[%.4f], recon y_fused=[%.4f], kl=[%.4f], latent z=[%.4f], sim s=[%.4f], sim z=[%.4f], adv s=[%.4f], adv s d=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon_x.item(), loss_recon_x_mix.item(), loss_recon_y.item(), loss_recon_y_fused.item(), loss_kl.item(), loss_latent_z.item(), loss_sim_s.item(), loss_sim_z.item(), loss_adv_s.item(), loss_adv_s_d.item()))

            # if iter > 1:
            #     break
            del si_list
            del zi_list
            del zi_mean_list
            del zi_log_var_list
            del xi_fake_list
            del xi_fake_mix_list
            del inputs_list
            targets.detach()
            inputs.detach()

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        if config['lambda_recon_y'] == 0 or config['lambda_recon_y_fused'] == 0:
            monitor_metric = stat['recon_x_mix']
        else:
            monitor_metric = stat['recon_y_fused']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        if config['is_discrim_s']:
            state['optimizer_d_s'] = optimizer_d_s.state_dict()
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        elif set == 'test_dropoff':
            dataset = TestDropoffDataset(config['data_path'], config['fold'], sel_idx_list=[438, 450], block_size=config['block_size'], contrast_list=config['contrast_list'])
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        else:
            raise ValueError('Undefined loader')

    loss_all_dict = {'recon_y': 0., 'recon_y_fused': 0., 'recon_x': 0., 'recon_x_mix': 0., 'kl': 0., 'latent_z': 0., 'sim_s': 0., 'sim_z': 0., 'adv_s': 0., 'adv_s_d': 0.,'all': 0.}

    subj_id_list = []
    slice_idx_list = []
    input_list = []
    target_list = []
    y_fake_fused_list = []
    y_fake_list_list = []
    xi_fake_mix_list_list = []
    xi_fake_list_list = []
    mask_list = []
    s_list = []
    z_list = []
    z_list_find_all = []
    metrics_list_dict = {}

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if info == 'nearest_neighbour' or info == 'mean':
        h5_path = os.path.join(res_path, 'results_all.h5')
        h5_saved = h5py.File(h5_path, 'r')
        s_list_saved = torch.tensor(h5_saved['s_list'], device=config['device'])
        z_list_saved = torch.tensor(h5_saved['z_list'], device=config['device'])
        s_compact_saved = []
        for idx in range(len(config['contrast_list'])):
            si_compact_saved = model.compute_compact_s(s_list_saved[:,idx])
            s_compact_saved.append(si_compact_saved)
        num_subj_saved = s_list_saved.shape[0] // 115

    with torch.no_grad():
        for iter, sample in enumerate(loader, 0):
            subj_id = sample['subj_id']
            slice_idx = sample['slice_idx']
            inputs = sample['inputs'].to(config['device'], dtype=torch.float)
            inputs_list = []
            for idx in range(len(config['contrast_list'])):
                inputs_list.append(inputs[:, idx*(2*config['block_size']+1):(idx+1)*(2*config['block_size']+1)])

            targets = sample['targets'].to(config['device'], dtype=torch.float)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            mask_img = sample['mask_img'].to(config['device'], dtype=torch.float)

            # pred, _ = model(inputs_list)
            si_list = model.compute_anatomy_encoding(inputs_list, mask_img)
            zi_list, zi_mean_list, zi_log_var_list = model.compute_modality_encoding(inputs_list, si_list, phase='test')

            if info == 'nearest_neighbour' or info == 'mean':
                num_subj = iter*config['batch_size'] // (135 - 21 + 1)
                print(iter, num_subj)
                if num_subj > 2:
                    break

                search_idx_list = []
                for i in range(num_subj_saved):
                    if i == num_subj:
                        continue
                    for j in range(115):
                        search_idx_list.append(115*i+j)

                z_list_find = []
                for i in range(len(config['contrast_list'])): # assume missing contrast i, by comparing s from contrast j
                    zi_list_find = []
                    si_compact = model.compute_compact_s(si_list[np.abs(1-i)])
                    for b in range(inputs.shape[0]):
                        if info == 'nearest_neighbour':
                            z_find = model.compute_nearest_neighbour_z_by_s(s_compact_saved[np.abs(1-i)][search_idx_list], z_list_saved[search_idx_list, i], si_compact[b])
                        else:
                            z_find = model.compute_mean_z_by_s(z_list_saved[search_idx_list, i])
                        zi_list_find.append(z_find.unsqueeze(0))
                    z_list_find.append(torch.cat(zi_list_find, 0))

                xi_fake_list = model.reconstruct_input_si_zi(si_list, z_list_find)
                xi_fake_mix_list = model.reconstruct_input_si_zj(si_list, z_list_find)

            else:
                xi_fake_list = model.reconstruct_input_si_zi(si_list, zi_list)
                xi_fake_mix_list = model.reconstruct_input_si_zj(si_list, zi_list)

            # recon y
            if iter == 0:
                # to simplify, only compute y once, if no loss on recon y
                y_fake_list = model.reconstruct_output_si(si_list)
                y_fake_fused = model.reconstruct_output_si_fused(si_list, mask)
            else:
                if config['lambda_recon_y'] > 0:
                    y_fake_list = model.reconstruct_output_si(si_list)
                if config['lambda_recon_y_fused'] > 0:
                    y_fake_fused = model.reconstruct_output_si_fused(si_list, mask)

            loss = 0
            if config['lambda_recon_y'] > 0:
                if config['dataset_name'] == 'BraTS':
                    loss_recon_y = model.compute_segmentation_loss_y_list(targets, y_fake_list, mask)
                else:
                    loss_recon_y = model.compute_recon_loss_y_list(targets, y_fake_list, mask, p=config['p'])
                loss += config['lambda_recon_y'] * loss_recon_y
            else:
                loss_recon_y = torch.tensor(0.)
            if config['lambda_recon_y_fused'] > 0:
                # if targets.shape[0] != y_fake_fused.shape[0]:
                #     pdb.set_trace()
                if config['dataset_name'] == 'BraTS':
                    loss_recon_y_fused = model.compute_segmentation_loss_y(targets, y_fake_fused)
                else:
                    loss_recon_y_fused = model.compute_recon_loss_y(targets, y_fake_fused, p=config['p'])
                loss += config['lambda_recon_y_fused'] * loss_recon_y_fused
            else:
                loss_recon_y_fused = torch.tensor(0.)
            if config['lambda_recon_x'] > 0:
                loss_recon_x = model.compute_recon_loss_x_list(inputs_list, xi_fake_list, mask, p=config['p'])
                loss += config['lambda_recon_x']*loss_recon_x
            else:
                loss_recon_x = torch.tensor(0.)
            if config['lambda_recon_x_mix'] > 0:
                loss_recon_x_mix = model.compute_recon_loss_x_mix_list(inputs_list, xi_fake_mix_list, mask, p=config['p'])
                loss += config['lambda_recon_x_mix']*loss_recon_x_mix
            else:
                loss_recon_x_mix = torch.tensor(0.)
            if config['lambda_kl'] > 0:
                if config['is_distri_z']:
                    zi_prior_mean_list, zi_prior_log_var_list = model.compute_zi_prior_distribution(targets.shape[0], len(config['contrast_list']), config['device'])
                    loss_kl = model.compute_kl_loss_list_two_gaussian(zi_mean_list, zi_log_var_list, zi_prior_mean_list, zi_prior_log_var_list, mask)
                else:
                    loss_kl = model.compute_kl_loss_list_standard(zi_mean_list, zi_log_var_list, mask)
            else:
                loss_kl = torch.tensor(0.)
            if config['lambda_latent_z'] > 0:
                # cycle back x_fake
                si_list_new = model.compute_anatomy_encoding(xi_fake_list, mask_img)
                zi_list_new, zi_mean_list_new, zi_log_var_list_new = model.compute_modality_encoding(xi_fake_list, si_list_new, phase='test')
                loss_latent_z = model.compute_latent_z_loss(zi_mean_list, zi_mean_list_new, mask)
                loss += config['lambda_latent_z']*loss_latent_z
            else:
                loss_latent_z = torch.tensor(0.)
            if config['lambda_sim_s'] > 0:
                loss_sim_s = model.compute_similarity_s_loss(si_list, mask)
                loss += config['lambda_sim_s']*loss_sim_s
            else:
                loss_sim_s = torch.tensor(0.)
            if config['lambda_sim_z'] > 0:
                loss_sim_z = model.compute_similarity_z_loss(zi_list, mask)
                loss += config['lambda_sim_z']*loss_sim_z
            else:
                loss_sim_z = torch.tensor(0.)
            if config['lambda_adv_s'] > 0:
                loss_adv_s_d, loss_adv_s = model.compute_adversarial_loss(si_list, mask)
                loss += config['lambda_adv_s']*loss_adv_s
            else:
                loss_adv_s = torch.tensor(0.)
                loss_adv_s_d = torch.tensor(0.)

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['recon_y_fused'] += loss_recon_y_fused.item()
            loss_all_dict['recon_x'] += loss_recon_x.item()
            loss_all_dict['recon_x_mix'] += loss_recon_x_mix.item()
            loss_all_dict['kl'] += loss_kl.item()
            loss_all_dict['latent_z'] += loss_latent_z.item()
            loss_all_dict['sim_s'] += loss_sim_s.item()
            loss_all_dict['sim_z'] += loss_sim_z.item()
            loss_all_dict['adv_s'] += loss_adv_s.item()
            loss_all_dict['adv_s_d'] += loss_adv_s_d.item()
            loss_all_dict['all'] += loss.item()

            if config['lambda_recon_y'] == 0 and config['lambda_recon_y_fused'] == 0:
                inputs_mix = []
                for i in range(mask.shape[1]):
                    for j in range(mask.shape[1]):
                        if i == j:
                            continue
                        inputs_mix.append(inputs_list[j])
                inputs_mix = torch.cat(inputs_mix, 0)
                xi_fake_mix = torch.cat(xi_fake_mix_list, 0)
                metrics = compute_reconstruction_metrics(inputs_mix.detach().cpu().numpy(), xi_fake_mix.detach().cpu().numpy())
            else:
                if config['dataset_name'] == 'BraTS':
                    metrics = compute_segmentation_metrics(targets.detach().cpu().numpy(), y_fake_fused.detach().cpu().numpy())
                else:
                    metrics = compute_reconstruction_metrics(targets.detach().cpu().numpy(), y_fake_fused.detach().cpu().numpy())
            print(metrics)
            # pdb.set_trace()

            for key in metrics.keys():
                if key in metrics_list_dict.keys():
                    metrics_list_dict[key].extend(metrics[key])
                else:
                    metrics_list_dict[key] = metrics[key]
            # print(metrics)
            # pdb.set_trace()

            if phase == 'test' and save_res:
                input_list.append(inputs.detach().cpu().numpy())
                target_list.append(targets.detach().cpu().numpy())
                y_fake_fused_list.append(y_fake_fused.detach().cpu().numpy())
                y_fake_list_list.append(torch.stack(y_fake_list,1).cpu().numpy())
                xi_fake_list_list.append(torch.stack(xi_fake_list,1).detach().cpu().numpy())
                xi_fake_mix_list_list.append(torch.stack(xi_fake_mix_list,1).detach().cpu().numpy())
                subj_id_list.append(subj_id)
                slice_idx_list.append(slice_idx.detach().cpu().numpy())
                mask_list.append(mask.detach().cpu().numpy())
                # pdb.set_trace()
                s_list.append(torch.stack(si_list, 1).detach().cpu().numpy())
                z_list.append(torch.stack(zi_list, 1).detach().cpu().numpy())
                if info == 'nearest_neighbour' or info == 'mean':
                    z_list_find_all.append(torch.stack(z_list_find, 1).detach().cpu().numpy())
            #
            if iter > 500:
                break

    for key in loss_all_dict.keys():
        loss_all_dict[key] /= (iter + 1)

    for key in metrics_list_dict:
        loss_all_dict[key] = np.array(metrics_list_dict[key]).mean()

    if phase == 'test' and save_res:
        input_list = np.concatenate(input_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        slice_idx_list = np.concatenate(slice_idx_list, axis=0)
        subj_id_list = np.concatenate(subj_id_list, axis=0)
        y_fake_fused_list = np.concatenate(y_fake_fused_list, axis=0)
        y_fake_list_list = np.concatenate(y_fake_list_list, axis=0)
        xi_fake_mix_list_list = np.concatenate(xi_fake_mix_list_list, axis=0)
        xi_fake_list_list = np.concatenate(xi_fake_list_list, axis=0)
        mask_list = np.concatenate(mask_list, axis=0)
        # pdb.set_trace()
        s_list = np.concatenate(s_list, axis=0)
        z_list = np.concatenate(z_list, axis=0)

        # data_dict = {'subj_id':subj_id_list, 'slice_idx':slice_idx_list,
        #             'inputs':input_list, 'targets':target_list, 'mask': mask_list,
        #             'y_fake_fused':y_fake_fused_list, 'y_fake_list':y_fake_list_list,
        #             'xi_fake_mix':xi_fake_mix_list_list, 'xi_fake_list':xi_fake_list_list,
        #             'si': s_list, 'zi': z_list}
        # path = os.path.join(res_path, 'results_all.npy')
        # np.save(path, data_dict)
        # pdb.set_trace()
        path = os.path.join(res_path, 'results_all'+info+'.h5')
        h5_file = h5py.File(path, 'w')
        h5_file.create_dataset('subj_id', data=np.string_(subj_id_list))
        h5_file.create_dataset('slice_idx', data=slice_idx_list)
        h5_file.create_dataset('inputs', data=input_list)
        h5_file.create_dataset('targets', data=target_list)
        h5_file.create_dataset('mask', data=mask_list)
        h5_file.create_dataset('y_fake_fused', data=y_fake_fused_list)
        h5_file.create_dataset('y_fake_list', data=y_fake_list_list)
        h5_file.create_dataset('xi_fake_mix', data=xi_fake_mix_list_list)
        h5_file.create_dataset('xi_fake_list', data=xi_fake_list_list)
        h5_file.create_dataset('s_list', data=s_list)
        h5_file.create_dataset('z_list', data=z_list)
        if info == 'nearest_neighbour' or info == 'mean':
            z_list_find_all = np.concatenate(z_list_find_all, axis=0)
            h5_file.create_dataset('z_list_find_all', data=z_list_find_all)

    return loss_all_dict

if config['phase'] == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
    # stat = evaluate(phase='test', set='test', save_res=True, info='_zeroT1')
    # stat = evaluate(phase='test', set='test', save_res=True, info='_zeroFLAIR')
    # stat = evaluate(phase='test', set='test', save_res=True, info='_avgT1')
    # stat = evaluate(phase='test', set='test', save_res=True, info='_avgFLAIR')
    # stat = evaluate(phase='test', set='test', save_res=True, info='noT1')
    # stat = evaluate(phase='test', set='test', save_res=True, info='nearest_neighbour')
    # stat = evaluate(phase='test', set='test', save_res=True, info='mean')
    # stat = evaluate(phase='test', set='test_dropoff', save_res=True)
    # stat = evaluate(phase='test', set='train', save_res=True)
    print(stat)
