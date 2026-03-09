import sys, json, os, argparse, time
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch
from utils.get_loaders import get_train_val_cls_loaders, modify_dataset
from utils.evaluation import evaluate_multi_cls
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.sam import SAM


# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--csv_train', type=str, default='data/TOR_4cls_csvs/train_tor.csv', help='path to training data csv')
parser.add_argument('--data_path', type=str, default='/home/wkabir/Newlook_Project/Newlook_AVR_Project/Adrian_Wazi_AVR/data/cropped_images/', help='path to training images')
parser.add_argument('--model_name', type=str, default='bit_resnext50_1', help='architecture')
parser.add_argument('--n_classes', type=int, default=6, help='number of categories')
parser.add_argument('--loss_fn', type=str, default='ce', help='loss function (ce)')
parser.add_argument('--oversample', type=str, default='1/1/1/1/1/1', help='oversampling per-class proportions')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate') # adam -> 1e-4, 3e-4
parser.add_argument('--min_lr', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--cycle_lens', type=str, default='5/3', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer choice')
parser.add_argument('--pretrained_weights', type=str, default=False, help='start from pretrained weights (path to)') # FIX IT
parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--n_checkpoints', type=int, default=1, help='nr of best checkpoints to keep (defaults to 3)')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'mcc':
        return operator.gt, 0
    elif metric == 'kappa':
        return operator.gt, 0
    elif metric == 'f1':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_one_epoch(loader, model, criterion, scheduler=None, optimizer=None, assess=False):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train: model.train()
    else: model.eval()
    if assess:
        probs_all, preds_all, labels_all = [], [], []

    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels, _) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if train:  # only in training mode
                loss.backward()
                if isinstance(optimizer, SAM):
                    optimizer.first_step(zero_grad=True)
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()  # for grad_acc_steps=0, this is just loss
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            if assess:
                probs = logits.softmax(dim=1)
                preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
                probs_all.extend(probs.detach().cpu().numpy())
                preds_all.extend(preds)
                labels_all.extend(labels.cpu().numpy())

            # Compute running loss
            running_loss += loss.detach().item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(loss_lr="{:.4f}/{:.6f}".format(float(run_loss), get_lr(optimizer)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    if assess: return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all), run_loss
    return None, None, None, None


def train_one_cycle(train_loader, oversample, model, criterion, optimizer, scheduler, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    # prepare cycle:
    # 1) build oversampled train loader
    csv_train_path = train_loader.dataset.csv_path
    train_loader_MOD = modify_dataset(train_loader, csv_train_path=csv_train_path, keep_samples=oversample, see_classes=True)

    # 2) reset iteration counter
    scheduler.last_epoch = -1
    # 3) update number of iterations
    scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader_MOD)

    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
        if epoch == cycle_len-1: assess=True # only get probs/preds/labels on last cycle
        else: assess = False
        # modify dataset at each epoch
        train_loader_MOD = modify_dataset(train_loader, csv_train_path=csv_train_path, keep_samples=oversample, see_classes=False)
        tr_preds, tr_probs, tr_labels, tr_loss = \
            run_one_epoch(train_loader_MOD, model, criterion, optimizer=optimizer, scheduler=scheduler, assess=assess)

    return tr_preds, tr_probs, tr_labels, tr_loss

def train_model(model, optimizer, train_criterion, val_criterion, train_loader, val_loader,
                oversample, scheduler, metric, exp_path, n_checkpoints):

    n_cycles = len(scheduler.cycle_lens)
    best_loss, best_auc, best_k, best_mcc, best_f1, best_cycle, best_models = 10, 0, 0, 0, 0, 0, []
    is_better, best_monitoring_metric = compare_op(metric)
    greater_is_better = best_monitoring_metric == 0
    all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs = [], [], [], []
    all_tr_f1s, all_vl_f1s, all_tr_losses, all_vl_losses = [], [], [], []

    class_names = ['C{}'.format(i) for i in range(model.n_classes)]
    print(class_names)
    print_conf, text_file_train, text_file_val = False, None, None

    for cycle in range(n_cycles):
        print('\nCycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle
        _, _, _, _ = train_one_cycle(train_loader, oversample, model, train_criterion, optimizer, scheduler, cycle=cycle)


        with torch.no_grad():
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(train_loader, model, val_criterion, assess=True)
            vl_preds, vl_probs, vl_labels, vl_loss = run_one_epoch(val_loader, model, val_criterion, assess=True)

        if exp_path is not None:
            print_conf = True
            text_file_train = osp.join(exp_path,'performance_cycle_{}.txt'.format(str(cycle+1).zfill(2)))
            text_file_val = osp.join(exp_path, 'performance_cycle_{}.txt'.format(str(cycle + 1).zfill(2)))

        tr_auc, tr_k, tr_mcc, tr_f1, _, tr_auc_all, tr_f1_all = evaluate_multi_cls(tr_labels, tr_preds, tr_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_train, loss=tr_loss)
        vl_auc, vl_k, vl_mcc, vl_f1, _, vl_auc_all, vl_f1_all = evaluate_multi_cls(vl_labels, vl_preds, vl_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_val, loss=vl_loss)

        print('Train||Val Loss: {:.4f}||{:.4f} - mAUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f} - F1: {:.2f}||{:.2f} - K: {:.2f}||{:.2f}'.format(
                                    tr_loss, vl_loss, 100*tr_auc, 100*vl_auc, 100*tr_mcc, 100*vl_mcc, 100*tr_f1, 100*vl_f1, 100*tr_k, 100*vl_k))

        if model.n_classes == 4:
            print('AUC: 0={:.2f}|{:.2f} - 1={:.2f}|{:.2f} - 2={:.2f}|{:.2f} - 3={:.2f}|{:.2f}'.format(100*tr_auc_all[0], 100*vl_auc_all[0],
                                                                                          100*tr_auc_all[1], 100*vl_auc_all[1],
                                                                                          100*tr_auc_all[2], 100*vl_auc_all[2],100*tr_auc_all[3], 100*vl_auc_all[3]))
        if model.n_classes == 3:
            print('AUC: DR0={:.2f}|{:.2f} - DR1={:.2f}|{:.2f} - DR2={:.2f}|{:.2f}'.format(100*tr_auc_all[0], 100*vl_auc_all[0],
                                                                                          100*tr_auc_all[1], 100*vl_auc_all[1],
                                                                                          100*tr_auc_all[2], 100*vl_auc_all[2]))
        if model.n_classes == 6:
            print('AUC: DR0={:.2f}|{:.2f} - DR1={:.2f}|{:.2f} - DR2={:.2f}|{:.2f} - DR3={:.2f}|{:.2f} - DR4={:.2f}|{:.2f} - U={:.2f}|{:.2f}'.format(
                    100 * tr_auc_all[0], 100 * vl_auc_all[0], 100 * tr_auc_all[1], 100 * vl_auc_all[1],
                    100 * tr_auc_all[2], 100 * vl_auc_all[2], 100 * tr_auc_all[3], 100 * vl_auc_all[3],
                    100 * tr_auc_all[4], 100 * vl_auc_all[4], 100 * tr_auc_all[5], 100 * vl_auc_all[5]))
## WAZIHA Debug
        if model.n_classes == 5:
            print('AUC: 0={:.2f}|{:.2f} - 1={:.2f}|{:.2f} - 2={:.2f}|{:.2f} - 3={:.2f}|{:.2f} - 4={:.2f}|{:.2f}'.format(
                    100 * tr_auc_all[0], 100 * vl_auc_all[0], 100 * tr_auc_all[1], 100 * vl_auc_all[1],
                    100 * tr_auc_all[2], 100 * vl_auc_all[2], 100 * tr_auc_all[3], 100 * vl_auc_all[3],
                    100 * tr_auc_all[4], 100 * vl_auc_all[4]))
## WAZIHA Debug

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'loss':  monitoring_metric = vl_loss
        elif metric == 'kappa': monitoring_metric = vl_k
        elif metric == 'mcc': monitoring_metric = vl_mcc
        elif metric == 'f1':  monitoring_metric = vl_f1
        elif metric == 'auc': monitoring_metric = vl_auc

        all_tr_aucs.append(tr_auc_all)
        all_vl_aucs.append(vl_auc_all)
        all_tr_mccs.append(tr_mcc)
        all_vl_mccs.append(vl_mcc)
        all_tr_f1s.append(tr_f1_all)
        all_vl_f1s.append(vl_f1_all)
        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)

        if is_better(monitoring_metric, best_monitoring_metric):
            print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, 100*best_monitoring_metric, 100*monitoring_metric))
            best_loss, best_k, best_mcc, best_f1, best_auc, best_cycle = vl_loss, vl_k, vl_mcc, vl_f1, vl_auc, cycle+1
            best_monitoring_metric = monitoring_metric
        else:
            print('-------- Best {} so far {:.2f} at cycle {:d} --------'.format(metric, 100 * best_monitoring_metric, best_cycle))


        # SAVE n best - keep deleting worse ones
        from operator import itemgetter
        import shutil
        if exp_path is not None:
            s_name = 'cycle_{}_mAUC_{:.2f}_MCC_{:.2f}_F1_{:.2f}'.format(str(cycle + 1).zfill(2),100*vl_auc, 100*vl_mcc, 100*vl_f1)
            best_models.append([osp.join(exp_path, s_name), monitoring_metric])

            if cycle < n_checkpoints:  # first n_checkpoints epochs save always
                print('-------- Checkpointing to {}/ --------'.format(s_name))
                save_model(osp.join(exp_path, s_name), model, optimizer, weights=True)
            else:
                worst_model = sorted(best_models, key=itemgetter(1), reverse=greater_is_better)[-1][0]# False for Loss, True for K
                if s_name != worst_model:  # this model was better than one of the best n_checkpoints models, remove that one
                    print('-------- Checkpointing to {}/ --------'.format(s_name))
                    save_model(osp.join(exp_path, s_name), model, optimizer, weights=True)
                    # print('before deleting', os.listdir(osp.join(exp_path, s_name)))
                    print('----------- Deleting {}/ -----------'.format(worst_model.split('/')[-1]))
                    shutil.rmtree(worst_model)
                    best_models = sorted(best_models, key=itemgetter(1), reverse=greater_is_better)[:n_checkpoints]

    del model
    torch.cuda.empty_cache()
    return best_auc, best_mcc, best_f1, all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs, \
                   all_tr_f1s, all_vl_f1s, all_tr_losses, all_vl_losses, best_cycle



if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    n_classes = args.n_classes
    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    max_lr, min_lr, bs = args.max_lr, args.min_lr, args.batch_size
    pretrained_weights = str2bool(args.pretrained_weights)

    oversample = args.oversample.split('/')
    oversample = list(map(float, oversample))
    if len(oversample) ==1: oversample = int(oversample[0])
    elif len(oversample) != n_classes: sys.exit('oversample must be a number or a tuple of len {:d}'.format(n_classes))

    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens)==2: # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0]*[cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)
        n_checkpoints = args.n_checkpoints
        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path, n_checkpoints=None, 0

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')
    data_path = args.data_path

    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_cls_loaders(csv_path_train=csv_train, csv_path_val=csv_val,
                                                         data_path=data_path, batch_size=bs,
                                                         tg_size=tg_size, mean=mean, std=std,
                                                         num_workers=args.num_workers)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, weight_decay=args.wd)
    elif optimizer_choice == 'sgd_sam':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=max_lr, momentum=0, weight_decay=args.wd)
    elif optimizer_choice == 'adam_sam':
        base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=max_lr)
    else:
        sys.exit('please choose a valid optimizer')

    # Example of how to start from pretrained weights
    if pretrained_weights is True:
        weights_path = osp.join('experiments/cyclical/bit50_sam/cycle_10_mAUC_94.18_MCC_65.44_F1_55.68/')
        try:
            model, stats, optimizer_state_dict = load_model(model, weights_path, device=device, with_opt=True)
            optimizer.load_state_dict(optimizer_state_dict)
            print('* Pretrained weights and optimizer loaded')
        except:
            sys.exit('Pretrained weights or optimizer not compatible for this model')
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = max_lr

    scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=min_lr)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    train_criterion, val_criterion = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()

    print('* Instantiating loss function', str(train_criterion))
    print('* Starting to train\n','-' * 10)
    start = time.time()
    b_mauc, b_mcc, b_f1, tr_aucs, vl_aucs, \
    tr_mccs, vl_mccs, tr_f1s, vl_f1s, tr_ls, vl_ls, b_cycle=train_model(model, optimizer, train_criterion, val_criterion,
                                                                 train_loader, val_loader, oversample, scheduler, metric,
                                                                 experiment_path, n_checkpoints)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("b_mauc: %f" % b_mauc)
    print("b_mcc: %f" % b_mcc)
    print("b_f1: %f" % b_f1)
    print("b_cycle: %d" % b_cycle)

    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best mAUC = {:.2f}\nBest MCC = {:.2f}\nBest F1 = {:.2f}\nBest cycle = {}\n'.format(
                100 * b_mauc, 100 * b_mcc, 100 * b_f1, b_cycle), file=f)
            for j in range(len(vl_aucs)):
                print('Cycle = {} -> mAUC={:.2f}/{:.2f}, '.format(j+1, 100*np.mean(tr_aucs[j]), 100*np.mean(vl_aucs[j])), file=f)
                if model.n_classes == 3:
                    print('AUC: noDME={:.2f}/{:.2f}, cDME={:.2f}/{:.2f}, DME={:.2f}/{:.2f} -- Loss={:.4f}/{:.4f}'.format(
                            100 * tr_aucs[j][0], 100 * vl_aucs[j][0], 100 * tr_aucs[j][1], 100 * vl_aucs[j][1],
                            100 * tr_aucs[j][2], 100 * vl_aucs[j][2], tr_ls[j], vl_ls[j]), file=f)
                elif model.n_classes == 4:
                    print('AUC: 0={:.2f}/{:.2f}, 1={:.2f}/{:.2f}, 2={:.2f}/{:.2f}, 3={:.2f}/{:.2f}'.format(
                            100 * tr_aucs[j][0], 100 * vl_aucs[j][0], 100 * tr_aucs[j][1], 100 * vl_aucs[j][1],
                            100 * tr_aucs[j][2], 100 * vl_aucs[j][2], 100 * tr_aucs[j][3], 100 * vl_aucs[j][3], tr_ls[j], vl_ls[j]), file=f)
                elif model.n_classes==6:
                    print('AUC: DR0={:.2f}/{:.2f}, DR1={:.2f}/{:.2f}, DR2={:.2f}/{:.2f}, '
                          'DR3={:.2f}/{:.2f}, DR4={:.2f}/{:.2f}, U={:.2f}/{:.2f}'.format(
                           100*tr_aucs[j][0], 100*vl_aucs[j][0], 100*tr_aucs[j][1], 100*vl_aucs[j][1], 100*tr_aucs[j][2], 100*vl_aucs[j][2],
                           100*tr_aucs[j][3], 100*vl_aucs[j][3], 100*tr_aucs[j][4], 100*vl_aucs[j][4], 100*tr_aucs[j][5], 100*vl_aucs[j][5]), file=f)
## Waziha Debug
                elif model.n_classes==5:
                    print('AUC: DR0={:.2f}/{:.2f}, DR1={:.2f}/{:.2f}, DR2={:.2f}/{:.2f}, '
                          'DR3={:.2f}/{:.2f}, DR4={:.2f}/{:.2f}'.format(
                           100*tr_aucs[j][0], 100*vl_aucs[j][0], 100*tr_aucs[j][1], 100*vl_aucs[j][1], 100*tr_aucs[j][2], 100*vl_aucs[j][2],
                           100*tr_aucs[j][3], 100*vl_aucs[j][3], 100*tr_aucs[j][4], 100*vl_aucs[j][4]), file=f)
## Waziha Debug

            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
