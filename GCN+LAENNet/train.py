import argparse
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import time
import datetime
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import *
from model import GCN
from pytorchtools import EarlyStopping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--train_percentage', type=float, default=0.)
    parser.add_argument('--eval_percentage', type=float, default=0.)
    parser.add_argument('--noise_fea_dim', type=int, default=0)
    parser.add_argument('--important_del_dim', type=int, default=0)
    parser.add_argument('--feature_noise', type=float, default=0.)
    parser.add_argument('--label_nr', type=float, default=0.)
    parser.add_argument('--nodes_per_class', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--pseudo_confidence', type=float, default=1.)
    parser.add_argument('--confidence_rate', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--file_redirect", type=str, default="suc")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument('--log_name', type=str)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    t_total = time.time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        adj, features, labels = load_data(args.dataset, args.noise_fea_dim, args.important_del_dim, args.feature_noise, args.adj_noise)
    else:
        adj, features, labels = load_npz(args.dataset, args.important_del_dim, args.feature_noise)
    # labels, train_indexes, eval_indexes, test_indexes, y_train, connected_nodes, eval_test_indexes = data_partion(args.dataset, labels, args.train_percentage, args.eval_percentage)
    labels, train_indexes, eval_indexes, test_indexes, y_train, connected_nodes, eval_test_indexes = data_partion_gcn_style(args.dataset, labels, args.nodes_per_class, args.label_nr)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                   factor=args.learning_rate_decay_factor,
                                                                   patience=args.learning_rate_decay_patience)
    diagnal = torch.eye(adj.shape[0])

    # Viewing.
    writer = SummaryWriter()

    # Early stopping
    early_stopping = EarlyStopping(args.patience, verbose=True)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        y_train = y_train.cuda()

    no_real_label_node_index = del_list(connected_nodes, train_indexes)
    current_labeled_node_index = copy.deepcopy(train_indexes)
    train_predict_flag = False
    eval_predict_flag = False
    test_predict_flag = False
    eval_test_predict_flag = False

    # Logging.
    path = 'log/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logname = os.path.join(path, current_time + args.file_redirect)
    initLogging(logname)

    best_tets_acc = 0.
    best_acc_epoch = 0
    epoch_adopted_pseudo_label_indexes = []
    pseudo_right = []
    for epoch in range(args.epochs):
        t_epoch = time.time()
        optimizer.zero_grad()
        # Current epoch labeled nodes.
        epoch_labeled_node_index = copy.deepcopy(current_labeled_node_index)
        logging.info("current_labeled_nodes_num: {}".format(len(current_labeled_node_index)))
        # Current epoch trained nodes, including train nodes and adopted pseudo labeled nodes.
        model.train()
        # Determine current unlabeled nodes in connected nodes and delete them from other nodes' neighborhood.
        if len(epoch_labeled_node_index) != len(connected_nodes):
            zero_indexes = del_list(connected_nodes, epoch_labeled_node_index)
            epoch_adj_y = obtain_adj(adj, zero_indexes)
            epoch_adj_x = torch.add(epoch_adj_y, diagnal)
            epoch_adj_y = sparse_mx_to_torch_sparse_tensor(normalize(sp.csr_matrix(epoch_adj_y.detach().numpy()))).cuda()
            #epoch_adj_y = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(epoch_adj_y.detach().numpy())).cuda()
            epoch_adj_x = sparse_mx_to_torch_sparse_tensor(normalize(sp.csr_matrix(epoch_adj_x.detach().numpy()))).cuda()
        # Perform one prediction.
        pred, _, _ = model(features, y_train, epoch_adj_x, epoch_adj_y)
        loss_train = F.nll_loss(pred[train_indexes], labels[train_indexes])

        model.eval()
        # Predict no real label node.
        t_predict = time.time()
        eval_pred, softmax_eval_pred, _ = model(features, y_train, epoch_adj_x, epoch_adj_y)
        loss_eval = F.nll_loss(eval_pred[eval_indexes], labels[eval_indexes])
        learning_rate_scheduler.step(loss_eval)
        # writer.add_scalar('eval/loss', loss_eval, epoch)

        # Collect the labeled neighborhood of all no real label nodes.
        if len(epoch_labeled_node_index) != len(connected_nodes):
            nrl_nodes_and_labeled_neighbors = obtain_no_real_label_node_labeled_neighbors(no_real_label_node_index, adj,
                                                                                      epoch_labeled_node_index)
            label_predicted_nrl_nodes = [int(key) for key in nrl_nodes_and_labeled_neighbors.keys()]
        # Obtain the pseudo label node indexes and update the node label matrix.
        hc_indexes = hc_obtain_predicted_label(softmax_eval_pred, args.confidence_rate, label_predicted_nrl_nodes)
        if len(epoch_labeled_node_index) != len(connected_nodes):
            current_labeled_node_index = train_indexes + hc_indexes
            current_labeled_node_index = sorted(current_labeled_node_index)
        y_train = update_y_train(y_train, torch.FloatTensor(softmax_eval_pred.cpu().detach().numpy()), hc_indexes)
        logging.info(
            "Epoch: {} predict {} no real node time:{}".format(epoch, len(hc_indexes), time.time() - t_predict))

        # Determine if the model can currently predict train/validation/test set accuracy.
        train_acc = accuracy(eval_pred[train_indexes], labels[train_indexes])
        logging.info("Epoch: {}, Train_loss: {}, Train_accuracy: {}".format(epoch, loss_train, train_acc))

        eval_acc = accuracy(eval_pred[eval_indexes], labels[eval_indexes])
        logging.info("Epoch: {}, Evaluate_loss: {}, Evaluate_accuracy: {}".format(epoch, loss_eval, eval_acc))

        test_acc = accuracy(eval_pred[test_indexes], labels[test_indexes])
        logging.info("Epoch: {}, Test_accuracy: {}".format(epoch, test_acc))
        if test_acc >= best_tets_acc:
            best_tets_acc = test_acc
            best_acc_epoch = epoch

        early_stopping(loss_eval, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        loss_train.backward()
        optimizer.step()
        # scheduler.step()
        logging.info("Best test accuracy is: {} at epoch {}".format(best_tets_acc, best_acc_epoch))
        logging.info("Epoch: {} time:{}".format(epoch, time.time() - t_epoch))
        print('')

    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    eval_pred, softmax_eval_pred, logit = model(features, y_train, epoch_adj_x, epoch_adj_y)
    test_acc = accuracy(eval_pred[test_indexes], labels[test_indexes])

    logging.info('Early stopping test accuracy: {}'.format(test_acc))
    logging.info('Best test accuracy: {} at epoch {}'.format(best_tets_acc, best_acc_epoch))
    logging.info('Optimization finished!')
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    logging.info('dataset: {}, dropout: {}, lr: {}, weight_decay: {}'.format(args.dataset,
                                                                             args.dropout,
                                                                             args.lr,
                                                                             args.weight_decay))

    logging.info('train_percentage: {}, eval_percentage: {}, nodes_per_class: {}'.format(args.train_percentage,
                                                                                         args.eval_percentage,
                                                                                         args.nodes_per_class))

    logging.info('important_del_dim: {}, feature_noise: {}, label_nr: {}'.format(args.important_del_dim,
                                                                                   args.feature_noise,
                                                                                   args.label_nr))

    logging.info('pseudo_confidence: {}, confidence_rate: {}'.format(args.pseudo_confidence,
                                                                     args.confidence_rate))
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    str1 = current_time + '-' + 'wd: ' + str(args.weight_decay) + '-' + 'lr: ' + str(args.lr) + '-' \
           + 'dropout: ' + str(args.dropout) + '-' + 'final_test_acc: ' + str(best_tets_acc) + \
           'tzqs: ' + str(args.important_del_dim) + '-' + 'tzzy: ' + str(args.feature_noise) + \
           '-' + 'bqzy: ' + str(args.label_nr) + '-' + 'bqsl: ' + str(args.nodes_per_class)
    file_name = 'sh_result/' + args.dataset + '-' + args.log_name
    with open(file_name, 'a') as file:
        file.write(str1 + '\n')