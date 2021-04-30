import os
import glob
import json
import math
from argparse import ArgumentParser
import torch.utils.data
import torch.nn as nn
from model.models import LocalizerNetwork, build_optimizer_and_scheduler
from utils.prepare_dataset import prepare_datasets
from utils.data_utils import get_data_loader
from utils.runner_utils import set_random_seed, convert_length_to_mask, evaluate_predictor

TACOS = 'tacos'
CHARADES = 'charades'

DATASET = TACOS

parser = ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets/', help='path to save processed dataset')
parser.add_argument('--task', type=str, default=DATASET, help='target task')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
# model parameters
parser.add_argument('--num_words', type=int, default=None, help='word dictionary size')
parser.add_argument('--num_chars', type=int, default=None, help='character dictionary size')
parser.add_argument('--word_dim', type=int, default=300, help='word embedding dimension')
parser.add_argument('--char_dim', type=int, default=100, help='character embedding dimension')
parser.add_argument('--visual_dim', type=int, default=4096, help='video feature dimension [i3d: 1024 | c3d: 4096]')
parser.add_argument('--dim', type=int, default=128, help='hidden size of the model')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads in transformer block')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
# training/evaluation parameters
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--mode', type=str, default='train', help='[train | test]')
parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
parser.add_argument('--model_dir', type=str, default='ckpt', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='Localizer', help='model name')
parser.add_argument('--epochs', type=int, default=45, help='maximal training epochs')
parser.add_argument('--num_train_steps', type=int, default=None, help='maximal training steps')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--init_lr", type=float, default=0.0004, help="initial learning rate") #0.0005
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument("--eval_period", type=int, default=300, help="evaluation period")
#video and query
parser.add_argument('--extend', type=float, default=0.1, help='extend rate')
parser.add_argument('--query_lambda', type=float, default=5.0, help='query high light lamda')
parser.add_argument('--sentence_encoder', type=str, default='base', help='query encoder')
parser.add_argument('--video_encoder', type=str, default='base', help='video encoder')
parser.add_argument('--video_encoder_drop', type=float, default=0.2, help='dropout rate')
parser.add_argument('--video_atte_heads', type=int, default=4, help='dropout rate')
parser.add_argument('--video_atte_layers', type=int, default=2, help='dropout rate')
parser.add_argument('--pos_freeze', action='store_true', help='use fixed positional embedding')
# contrastive learning
parser.add_argument('--vq_mi_lambda', type=float, default=1, help='loss weight')
parser.add_argument('--vv_mi_lambda', type=float, default=1, help='loss weight')
parser.add_argument('--vq_mi', dest='vq_mi', action='store_true', help='mutual information for video and query.')
parser.add_argument('--vv_mi', dest='vv_mi', action='store_true', help='mutual between start point and other clips.')

cfgs = parser.parse_args()
print("Dataset:", cfgs.task)
print(cfgs)
# set random seed
set_random_seed(cfgs.seed)

# check if dataset is processed
data_path = os.path.join(cfgs.save_dir, '_'.join([cfgs.task, str(cfgs.max_pos_len)]) + '.pkl')
if not os.path.exists(data_path):
    prepare_datasets(cfgs)

# Device configuration
cuda_str = 'cuda' if cfgs.gpu_idx is None else 'cuda:{}'.format(cfgs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# preset model save dir
model_dir = os.path.join(cfgs.model_dir, '_'.join([cfgs.model_name.lower(), cfgs.task, str(cfgs.max_pos_len), str(cfgs.query_lambda)]))

if cfgs.mode == 'train':
    train_loader, test_loader, num_words, num_chars, word_vectors, train_samples, test_samples = get_data_loader(cfgs)
    cfgs.num_words = num_words
    cfgs.num_chars = num_chars
    cfgs.num_train_steps = math.ceil(train_samples / cfgs.batch_size) * cfgs.epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'configs.json'), mode='w', encoding='utf-8') as f:
        json.dump(vars(cfgs), f, indent=4, sort_keys=True)
    # build model
        model = LocalizerNetwork(cfgs=cfgs, word_vectors=word_vectors).to(device)
    optimizer,scheduler = build_optimizer_and_scheduler(model, cfgs=cfgs)

    best_r1i7 = -1.0
    best_r1i5 = -1.0
    best_r1i3 = -1.0
    best_r1i1 = -1.0
    best_m = -1.0
    best_epoch = -1
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    step_counter = 0
    total_loss = 0
    for epoch in range(cfgs.epochs):
        model.train()
        for video_features, word_ids, char_ids, _, _, s_labels, e_labels, num_units, _, hightlight_labels, pos, ner, deprel, head, words in train_loader:
            step_counter += 1
            # prepare features
            video_features, num_units = video_features.to(device), num_units.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels = s_labels.to(device), e_labels.to(device)
            hightlight_labels = hightlight_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(num_units, cfgs.max_pos_len).to(device)
            # compute logits and loss
            start_logits, end_logits, vq_mi_loss, vv_mi_loss = model(word_ids, char_ids, video_features, video_mask, query_mask, hightlight_labels, s_labels, e_labels, pos, ner, deprel, head,words, is_training=True)
            loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)

            if cfgs.vq_mi:
                loc_loss += cfgs.vq_mi_lambda * vq_mi_loss
            if cfgs.vv_mi:
                loc_loss += cfgs.vv_mi_lambda * vv_mi_loss
                
            optimizer.zero_grad()
            loc_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfgs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            total_loss += loc_loss.item()
            if step_counter % cfgs.period == 0:
                print('epoch: %d | step: %d | lr: %.6f| loss | loc: %.6f | total loss:: %.6f' % (epoch + 1, step_counter, optimizer.param_groups[0]['lr'],loc_loss.item(),total_loss),
                      flush=True)
                total_loss = 0
            if (cfgs.task == "tacos" and epoch > 5)  or (cfgs.task == "charades" and epoch > 15):
                if step_counter % cfgs.eval_period == 0:
                    model.eval()
                    r1i1, r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, cfgs)
                    print('\nepoch: %d | step: %d | evaluation (Rank@1) | IoU=0.1: %.2f | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
                          'mIoU: %.2f\n' % (epoch + 1, step_counter, r1i1, r1i3, r1i5, r1i7, mi), flush=True)
                    score_str = 'epoch: %d | step: %d | Rank@1 | IoU=0.1: %.2f | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | mIoU: ' \
                                '%.2f\n' % (epoch + 1, step_counter, r1i1, r1i3, r1i5, r1i7, mi)
                    score_writer.write(score_str)
                    score_writer.flush()
                    if r1i7 > best_r1i7:
                        best_r1i7 = r1i7
                        best_r1i5 = r1i5
                        best_r1i3 = r1i3
                        best_r1i1 = r1i1
                        best_m = mi
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(model_dir, '{}_epoch_{}_step_{}_model_{:.2f}.t7'.format(
                            cfgs.model_name, epoch + 1, step_counter, r1i7)))
                    model.train()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr > 0.0001:
            current_lr = current_lr * 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        if (cfgs.task == "tacos" and epoch > 5)  or (cfgs.task == "charades" and epoch > 5):
            model.eval()
            r1i1, r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, cfgs)
            print('\nepoch: %d | step: %d | evaluation (Rank@1) | IoU=0.1: %.2f | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
                  'mIoU: %.2f\n' % (epoch + 1, step_counter, r1i1, r1i3, r1i5, r1i7, mi), flush=True)
            score_str = 'epoch: %d | step: %d | Rank@1 | IoU=0.1: %.2f | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | mIoU: ' \
                        '%.2f\n' % (epoch + 1, step_counter, r1i1, r1i3, r1i5, r1i7, mi)
            score_writer.write(score_str)
            score_writer.flush()
            if r1i7 > best_r1i7:
                best_r1i7 = r1i7
                best_r1i5 = r1i5
                best_r1i3 = r1i3
                best_r1i1 = r1i1
                best_m = mi
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(model_dir, '{}_epoch_{}_step_{}_model_{:.2f}.t7'.format(
                    cfgs.model_name, epoch + 1, step_counter, r1i7)))

            print('Best score: Epoch: %d | IoU=0.1: %.2f | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
                  'mIoU: %.2f\n' % (best_epoch + 1, best_r1i1, best_r1i3, best_r1i5, best_r1i7, best_m), flush=True)
    score_writer.close()
        

elif cfgs.mode == 'test':
    print("Testing")
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    with open(os.path.join(model_dir, 'configs.json'), mode='r', encoding='utf-8') as f:
        pre_configs = json.load(f)
    parser.set_defaults(**pre_configs)
    cfgs = parser.parse_args()
    # load dataset
    _, test_loader, _, _, word_vectors, _, test_samples = get_data_loader(cfgs)
    # build model
    model = LocalizerNetwork(cfgs=cfgs, word_vectors=word_vectors).to(device)
    # testing
    filenames = glob.glob(os.path.join(model_dir, '*.t7'))
    filenames.sort()

    # old testing
    # for filename in filenames:
    #     epoch = int(os.path.basename(filename).split('_')[2])
    #     step = int(os.path.basename(filename).split('_')[4])
    #     model.load_state_dict(torch.load(filename))
    #     model.eval()
    #     r1i1, r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, cfgs)
    #     print('epoch: %d | step: %d | evaluation (Rank@1) | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
    #           'mIoU: %.2f' % (epoch + 1, step, r1i3, r1i5, r1i7, mi), flush=True)

    best_score = 0
    file_idx = 0
    epoch = 0
    step = 0
    for i in range(len(filenames)):
        filename = filenames[i]
        score = float(os.path.basename(filename).split('_')[-1].replace('.t7', ''))
        epochi = int(os.path.basename(filename).split('_')[2])
        stepi = int(os.path.basename(filename).split('_')[4])
        if score > best_score:
            best_score = score
            file_idx = i 
            epoch = epochi 
            step = stepi
    
    filename = filenames[file_idx]
    print("Best model file:", filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    r1i1, r1i3, r1i5, r1i7, mi = evaluate_predictor(model, test_loader, device, epoch, test_samples, cfgs)
    print('epoch: %d | step: %d | evaluation (Rank@1) | IoU=0.3: %.2f | IoU=0.5: %.2f | IoU=0.7: %.2f | '
            'mIoU: %.2f' % (epoch + 1, step, r1i3, r1i5, r1i7, mi), flush=True)

else:
    raise ValueError('Unknown mode, only support [train | test]!!!')
