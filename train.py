import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import argparse

import dataloader_prompt_add
import dataloader_images as dataloader_sharp 

import model_small
import numpy as np

from test_function import inference

import clip_score
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import clip

import pyiqa
import shutil


def setupseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # cudnn


setupseed(42)

task_name = "train0"
writer = SummaryWriter('./' + task_name+"/" + 'tensorboard_' + task_name)

dstpath = "./" + task_name + "/" + "train_scripts"
if not os.path.exists(dstpath):
    os.makedirs(dstpath)
shutil.copy("train.py", dstpath)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running in the device:", device)
# load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")  # ViT-B/32
model.to(device)
for para in model.parameters():
    para.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # print("82 x.size() is :", x.size())  # torch.Size([2, 512])
        return x


class MLP(nn.Module):
    def __init__(self, sizes, dropout, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
#            layers.append(dropout)
#            if i == (len(sizes) - 2):  # #######################################
#                layers.append(act())
        layers.append(dropout)
        layers.append(act())
        self.clip_project = nn.Sequential(*layers)

    def forward(self, x):
        return 10 * self.clip_project(x)


# class MLP(nn.Module):
#     def __init__(self, sizes, dropout, bias=True):
#         super(MLP, self).__init__()
#         layers = []
#         for i in range(len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
#             # layers.append(dropout)
#             # if i == (len(sizes) - 2):  # #######################################
#             #     layers.append(act())
#         layers.append(dropout)
#         self.clip_project = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.clip_project(x)


def prompts(initials=None):
    text_encoder = TextEncoder(model)
    if isinstance(initials, list):
        # print("194 The initial prompts are list:", initials)
        text = clip.tokenize(initials).cuda()
        # print("196 text is :", text)
        # print("197 text.size() is :", text.size())  # torch.Size([2, 77])
        # self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        embedding_prompt = model.token_embedding(text)
    elif isinstance(initials, str):
        # print("The initial prompts are str:", initials)
        prompt_path = initials

        state_dict = torch.load(prompt_path)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        embedding_prompt = new_state_dict['embedding_prompt']
        # self.embedding_prompt.requires_grad = True
    else:
        # print("The initial prompts are :", initials)
        embedding_prompt = model.token_embedding([" ".join(["X"] * config.length_prompt),
                                                  " ".join(["X"] * config.length_prompt)])

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in [config.initial_well_prompts]])
    # print("218 tokenized_prompts.size() is :", tokenized_prompts.size())  # torch.Size([1, 77])
    # print("219 self.embedding_prompt.size() is :", embedding_prompt.size())  # torch.Size([2, 77, 512])
    text_features = text_encoder(embedding_prompt, tokenized_prompts)
    # print("221 text_features.size() is :", text_features.size())  # torch.Size([2, 512])
    return text_features


class Probs(nn.Module):
    def __init__(self):
        super(Probs, self).__init__()
        self.size = [512, 256, 512]
        self.dropout = nn.Dropout(0.5)
        self.text_mlp = MLP(self.size, self.dropout)

    def forward(self, tensor, flag=1):
        text_features = prompts(config.initial_prompts)
        # print("234 text_features.size() is :", text_features.size())  # torch.Size([2, 512])
        text_features = self.text_mlp(text_features)
        # print("236 text_features.size() is :", text_features.size())  # torch.Size([2, 512])
        # print("237 tensor.size() is :", tensor.size())  # torch.Size([16, 1, 512]); torch.Size([14, 1, 512])
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            # print("159 image_features.size() is :", image_features.size())  # torch.Size([1, 512])
            # print("160 image_features is :", image_features)  # torch.Size([1, 512])
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            # print("162 text_features is :", text_features)
            # print("163 text_features / nor.size() is :", (text_features / nor).size())  # torch.Size([1, 512])
            # print("164 text_features / nor is :", text_features / nor)  # torch.Size([1, 512])
            if flag == 0:
                similarity = (100.0 * image_features @ (text_features / nor).T)  # .softmax(dim=-1)
                if i == 0:
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features / nor).T).softmax(dim=-1)  # /nor
                if i == 0:
                    probs = similarity[:, 0]
                else:
                    probs = torch.cat([probs, similarity[:, 0]], dim=0)
        return probs


def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    # load model
    U_net = model_small.UNet_emb_oneBranch_symmetry_noreflect(3, 1).cuda()
  
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    # add pretrained model weights: prompts and enhancement
    if config.load_pretrain_prompt == True:
        # learn_prompt = Prompts(config.prompt_pretrain_dir).cuda()
        learn_prompt = Probs().load(config.prompt_pretrain_dir).cuda()
        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "pretrained_prompt" + '.pth')
    else:
        if config.num_clip_pretrained_iters < 3000:
            # print("WARNING: For training from scratch, num_clip_pretrained_iters should not lower than 3000 iterations!"
            #       "\nAutomatically reset num_clip_pretrained_iters to 8 iterations...")
            config.num_clip_pretrained_iters = 8  # 8000
        # learn_prompt = Prompts([" ".join(["X"] * (config.length_prompt)), " ".join(["X"] * (config.length_prompt))]).cuda()
        learn_prompt = Probs().cuda()
    learn_prompt = torch.nn.DataParallel(learn_prompt)
    U_net.apply(weights_init)
    
    if config.load_pretrain == True:
        print("The load_pretrain is True, thus num_reconstruction_iters is automatically set to 0.")
        config.num_reconstruction_iters = 0
        state_dict = torch.load(config.pretrain_dir)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        U_net.load_state_dict(new_state_dict)
        # U_net.load_state_dict(torch.load(config.pretrain_dir))
        torch.save(U_net.state_dict(), config.train_snapshots_folder + "pretrained_network" + '.pth')
    else:
        if config.num_reconstruction_iters < 200:
            # print("WARNING: For training from scratch, num_reconstruction_iters should not lower than 200 iterations!"
            #       "\nAutomatically reset num_reconstruction_iters to 6 iterations...")
            config.num_reconstruction_iters = 6  # ############1000################
    U_net = torch.nn.DataParallel(U_net)
    
    # load dataset
    train_dataset = dataloader_sharp.lowlight_loader(config.lowlight_images_path, config.overlight_images_path)  # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    prompt_train_dataset = dataloader_prompt_add.lowlight_loader(config.lowlight_images_path,
                                                                 config.normallight_images_path)
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size,
                                                      shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # loss
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    
    # load gradient update strategy.
    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=config.prompt_lr, weight_decay=config.weight_decay)

    # initial parameters
    U_net.train()
    cur_iteration = 0
    max_score_psnr = -10000
    pr_last_few_iter = 0
    score_psnr = [0]*30

    best_prompt = learn_prompt
    min_prompt_loss = 100
    reconstruction_iter = 0
    reinit_flag = 0

    print("######################## selecting the training stage: #######################################")
    # Start training!
    for epoch in range(config.num_epochs):
        # training the prompts
        if cur_iteration < config.num_clip_pretrained_iters:
            train_phrase = "prompts_learning"
            total_iteration = config.num_clip_pretrained_iters
            print("Training in the phrase of {}".format(train_phrase))

            for name, param in U_net.named_parameters():
                param.requires_grad_(False)

            for iteration, item in enumerate(prompt_train_loader):
                img_lowlight, label = item
                img_lowlight, label = img_lowlight.cuda(), label.cuda()
                # print("513 img_lowlight.size() is : ", img_lowlight.size())  # torch.Size([16, 1, 512])
                # print("514 img_lowlight is : ", img_lowlight)
                # print("515 label.size() is : ", label.size())  # torch.Size([16])
                # print("516 label is : ", label)  # tensor([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1], device)
                output = learn_prompt(img_lowlight, 0)
                # print("518 output: {}, label: {}".format(output, label))
                loss = 10 * F.cross_entropy(output, label)
                # print("348 loss: {}".format(loss))
                prompt_optimizer.zero_grad()
                loss.backward()
                prompt_optimizer.step()

                # 每prompt_display_iter次，比较loss值，tensorboard记录loss变化，并保存loss_min的prompts模型.pth。
                if ((cur_iteration + 1) % config.prompt_display_iter) == 0:
                    if loss < min_prompt_loss:
                        min_prompt_loss = loss
                        best_prompt = learn_prompt
                        # best_prompt_iter = cur_iteration + 1
                        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder +
                                   "best_prompt" + '.pth')
                    # print("prompt current learning rate is : ", prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", cur_iteration + 1, "/epoch ", epoch, "is :", loss.item())
                    # print("output :", output.softmax(dim=-1), "; label :", label)
                    # print("cross_entropy_loss : ", loss)
                    writer.add_scalars('Loss_prompt', {'train': loss}, cur_iteration)
                    print('cur_iteration :', cur_iteration + 1, " ", "total_iteration :", total_iteration)

                # 每prompt_snapshot_iter次，保存对应的prompts模型.pth，不一定是loss最min的pth。
                if ((cur_iteration + 1) % config.prompt_snapshot_iter) == 0:
                    torch.save(learn_prompt.state_dict(),
                               config.prompt_snapshots_folder + "iter_" + str(cur_iteration + 1) + '.pth')

                # 若prompt训练阶段完成后，loss依然大于预定值，即学习效果不好，则增加训练iteration数。
                if cur_iteration + 1 == total_iteration and loss > config.thre_prompt:  # loss>last_prompt_loss[flag_prompt]*0.95:#loss>0.01:#
                    print("cur_iteration: {}, total_iteration: {}".format(cur_iteration + 1, total_iteration))
                    print("loss: {},  config.thre_prompt: {}".format(loss, config.thre_prompt))
                    total_iteration += 1000  # ###################100#####################

                # 训练完成，跳出
                elif cur_iteration + 1 == total_iteration:
                    cur_iteration += 1
                    break
                cur_iteration += 1

        # training reconstruction
        elif cur_iteration < config.num_reconstruction_iters + config.num_clip_pretrained_iters:
            train_phrase = "reconstruction_learning"
            total_iteration = config.num_reconstruction_iters + config.num_clip_pretrained_iters
            print("Training in the phrase of {}".format(train_phrase))

            # fix the prompt and train the enhancement model
            for name, param in learn_prompt.named_parameters():
                param.requires_grad_(False)

            for name, param in U_net.named_parameters():
                param.requires_grad_(True)

            for iteration, item in enumerate(train_loader):
                img_lowlight, img_lowlight_path = item

                img_lowlight = img_lowlight.cuda()

                light_map = U_net(img_lowlight)
                final = torch.clamp((img_lowlight / (light_map + 0.000000001)), 0, 1)

                # ####################### reconstruction ##########################
                loss = 25 * L_clip_MSE(final, img_lowlight, [1.0, 1.0, 1.0, 1.0, 1.0])

                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                with torch.no_grad():
                    # print("switch to 418")
                    score_psnr[pr_last_few_iter] = torch.mean(iqa_metric(img_lowlight, final))
                    # print("420 img_lowlight.size() :", img_lowlight.size())
                    # print("421 final.size() :", final.size())
                    # print("422 score_psnr[{}] is: {}".format([pr_last_few_iter], score_psnr[pr_last_few_iter]))
                    # print("423 score_psnr is:", score_psnr)
                    reconstruction_iter += 1
                    # print("425 sum(score_psnr).item()/30.0 is :", sum(score_psnr).item() / 30.0)

                    # 若reconstruction100个iterations之后效果太差，则重新初始化训练
                    if sum(score_psnr).item() / 30.0 < 8 and reconstruction_iter > 100:
                        # print("switch to 429")
                        reinit_flag = 1

                    pr_last_few_iter += 1
                    if pr_last_few_iter == 30:
                        pr_last_few_iter = 0
                    if (sum(score_psnr).item() / 30.0) > max_score_psnr and ((cur_iteration + 1) % config.display_iter) == 0:
                        # print("switch to 436")
                        max_score_psnr = sum(score_psnr).item() / 30.0
                        torch.save(U_net.state_dict(), config.train_snapshots_folder + "best_model" + "_reconstruction_" +
                                   str(cur_iteration + 1) + '.pth')
                        best_model = U_net
                        best_model_iter = cur_iteration + 1
                        print("max_score_psnr is :", max_score_psnr)
                        ############################################################################################
                        # ################################ inference ###############################################
                        inference(config.lowlight_images_path,
                                  './' + task_name + '/result_' + task_name + '/result_jt_' +
                                  str(best_model_iter) + "_psnr_or_-loss" + str(max_score_psnr)[:8] + '/', best_model, 256)

                if reinit_flag == 1:
                    # print("457 sum(score_psnr).item()/30.0 is :", sum(score_psnr).item()/30.0)
                    print("reinitialization...")
                    seed = random.randint(0, 100000)
                    print("current random seed: ", seed)
                    torch.cuda.manual_seed_all(seed)
                    U_net = model_small.UNet_emb_oneBranch_symmetry_noreflect(3, 1).cuda()
                    U_net.apply(weights_init)
                    U_net = torch.nn.DataParallel(U_net)
                    reconstruction_iter = 0
                    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train_lr,
                                                       weight_decay=config.weight_decay)
                    total_iteration += 100
                    reinit_flag = 0

                if ((cur_iteration + 1) % config.display_iter) == 0:
                    # print("training current learning rate: ", train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration: ", cur_iteration + 1, "/epoch ", epoch, " loss is:", loss.item())
                    writer.add_scalars('Loss_train', {'train': loss}, cur_iteration + 1)
                    print("cur_iteration+1 :{}, total_iteration:{}".format(cur_iteration + 1, total_iteration))

                # 完成reconstruction learning，跳出
                if cur_iteration + 1 == total_iteration:
                    cur_iteration += 1
                    break
                cur_iteration += 1

        # training enhancement
        else:
            train_phrase = "enhancement learning"
            print("Training in the phrase of {}".format(train_phrase))

            if cur_iteration - (config.num_reconstruction_iters + config.num_clip_pretrained_iters) == 0:
                learn_prompt = best_prompt
                U_net = best_model

                max_score_psnr = -10000
                score_psnr = [0] * 30
                pr_last_few_iter = 0

            for name, param in U_net.named_parameters():
                param.requires_grad_(True)

            text_features = prompts(config.initial_prompts)
            text_features_embedding = learn_prompt.module.text_mlp
            text_features_embedding.require_grad = False
            text_features = text_features_embedding(text_features)

            # fix the prompt and train the enhancement model
            for name, param in learn_prompt.named_parameters():
                param.requires_grad_(False)

            for iteration, item in enumerate(train_loader):
                img_lowlight, img_lowlight_path = item

                img_lowlight = img_lowlight.cuda()

                light_map = U_net(img_lowlight)
                final = torch.clamp((img_lowlight / (light_map + 0.000000001)), 0, 1)

                cliploss = 16 * 20 * L_clip(final, text_features)
                clip_MSEloss = 25 * L_clip_MSE(final, img_lowlight, [1.0, 1.0, 1.0, 1.0, 0.5])

                # ####################### enhancement #############################
                loss = cliploss + 0.9 * clip_MSEloss

                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                with torch.no_grad():
                    score_psnr[pr_last_few_iter] = -loss

                    pr_last_few_iter += 1
                    if pr_last_few_iter == 30:
                        pr_last_few_iter = 0
                    if (sum(score_psnr).item() / 30.0) > max_score_psnr and (
                            (cur_iteration + 1) % config.display_iter) == 0:
                        max_score_psnr = sum(score_psnr).item() / 30.0
                        torch.save(U_net.state_dict(), config.train_snapshots_folder + "best_model" + "_enhancement_" +
                                   str(cur_iteration + 1) + '.pth')
                        best_model = U_net
                        best_model_iter = cur_iteration + 1
                        print("461 max_score_psnr is :", max_score_psnr)
                        ############################################################################################
                        # ################################ inference ###############################################
                        inference(config.lowlight_images_path,
                                  './' + task_name + '/result_' + task_name + '/result_jt_' +
                                  str(best_model_iter) + "_psnr_or_-loss" + str(max_score_psnr)[:8] + '/', best_model, 256)

                if ((cur_iteration + 1) % config.display_iter) == 0:
                    # print("training current learning rate: ", train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration: ", cur_iteration + 1, "/epoch ", epoch, "loss is:", loss.item())
                    # print("loss_clip: ", cliploss, " reconstruction loss: ", clip_MSEloss)
                    writer.add_scalars('Loss_train', {'train': loss, "clip": cliploss,
                                                      "reconstruction loss": clip_MSEloss}, cur_iteration + 1)

                cur_iteration += 1


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('-p', '--initial_prompts', type=list,
                        default=["a photo of a normal light scene", "a photo of a backlit scene"])
    parser.add_argument('-w', '--initial_well_prompts', type=str, default="a photo of a normal light scene")
    parser.add_argument('-b', '--lowlight_images_path', type=str,
                        default="/root/autodl-tmp/Dataset/Clip_LIT/train_data/BAID_380/resize_input/")
    parser.add_argument('--overlight_images_path', type=str, default=None)
    parser.add_argument('-r', '--normallight_images_path', type=str,
                        default='/root/autodl-tmp/Dataset/Clip_LIT/train_data/DIV2K_384/')
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--thre_train', type=float, default=90)
    parser.add_argument('--thre_prompt', type=float, default=60)
    parser.add_argument('--reconstruction_train_lr', type=float, default=0.00005)  # 0.0001
    parser.add_argument('--train_lr', type=float, default=0.00002)  # 0.00002#0.00005#0.0001
    # parser.add_argument('--prompt_lr', type=float, default=0.000005)  # 0.00001#0.00008
    parser.add_argument('--prompt_lr', type=float, default=0.00005)  # 0.00001#0.00008
    parser.add_argument('--T_max', type=float, default=100)
    parser.add_argument('--eta_min', type=float, default=5e-6)  # 1e-6
    parser.add_argument('--weight_decay', type=float, default=0.001)  # 0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    # parser.add_argument('--num_epochs', type=int, default=2000)  # 3000
    parser.add_argument('--num_epochs', type=int, default=3000)  # 3000 #############3000################
    # parser.add_argument('--num_reconstruction_iters', type=int, default=0)  # 1000
    parser.add_argument('--num_reconstruction_iters', type=int, default=2000)  # 1000 #############1000################
    # parser.add_argument('--num_clip_pretrained_iters', type=int, default=0)  # 8000
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=8000)  # 8000 #############8000################
    parser.add_argument('--noTV_epochs', type=int, default=100)  # ############100################
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--prompt_batch_size', type=int, default=16)  # 32
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=20)  # ############20################
    parser.add_argument('--snapshot_iter', type=int, default=20)  # ############20################
    parser.add_argument('--prompt_display_iter', type=int, default=20)  # ############20################
    parser.add_argument('--prompt_snapshot_iter', type=int, default=100)  # ############100################
    parser.add_argument('--train_snapshots_folder', type=str,
                        default="./" + task_name + "/" + "snapshots_train_" + task_name + "/")
    parser.add_argument('--prompt_snapshots_folder', type=str,
                        default="./" + task_name + "/" + "snapshots_prompt_" + task_name + "/")
    # parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--pretrain_dir', type=str,
                        default='./pretrained_models/init_pretrained_models/init_enhancement_model.pth')
    # parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default=False)
    # parser.add_argument('--prompt_pretrain_dir', type=str,
    #                     default='./pretrained_models/init_pretrained_models/init_prompt_pair.pth')
    parser.add_argument('--prompt_pretrain_dir', type=str, default=False)
    
    config = parser.parse_args()

    if not os.path.exists(config.train_snapshots_folder):
        os.mkdir(config.train_snapshots_folder)
    if not os.path.exists(config.prompt_snapshots_folder):
        os.mkdir(config.prompt_snapshots_folder)

    train(config)
