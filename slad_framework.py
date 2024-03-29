import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, auc
import torch.nn as nn
from tqdm import tqdm

from slad_dataloader import load_dataset
from slad_mcts import mcts
from slad_model import SLADGNN

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # fix model deterministic


def avg_min_dist(prototype1, prototype2):
    distances = torch.cdist(prototype1, prototype2)
    min_distances, min_indices = torch.min(distances, dim=-1)
    avg_min_distance = torch.mean(min_distances)
    return avg_min_distance


def compute_estimated_prototype_layer(train_prot_dataloader, rollout_num, min_atoms, k_hop_graphs, gnnNet,
                                      prototype_num_each_class, random_negative_sample, device, mcts_pruning_strategy,
                                      mcts_stop_strategy, k_neighbors, mcts_top_bound, mcts_prot_selection):
    gnnNet.eval()
    print("%-----estimated prototype layer-----%")
    output.append("%-----estimated prototype layer-----%")
    with torch.no_grad():
        estimated_prototype_layer = mcts(dataset_name, train_prot_dataloader, rollout_num, min_atoms, k_hop_graphs,
                                         gnnNet.GNN,
                                         prototype_num_each_class,
                                         random_negative_sample, device, mcts_pruning_strategy,
                                         mcts_stop_strategy, k_neighbors, mcts_top_bound, mcts_prot_selection)
    estimated_prototype_layer = estimated_prototype_layer.to(device)
    print("estimated prototype layer:")
    output.append("estimated prototype layer:")
    for new_prot_l in estimated_prototype_layer.tolist():
        print("%-----------------------------------%")
        output.append("%-----------------------------------%")
        for new_prot in new_prot_l:
            print(new_prot)
            output.append(str(new_prot))
    print("%-----------------------------------%")
    output.append("%-----------------------------------%")
    return estimated_prototype_layer


def prototype_loss_with_mcts(gnnNet, estimated_prototype_layer):
    gnnNet.eval()
    current_prototype_layer = gnnNet.prototype_layer
    mean_dist_list = []
    for i in range(current_prototype_layer.size(0)):
        estimated_layer_with_class = estimated_prototype_layer[i]
        current_layer_with_class = current_prototype_layer[i]
        mean_dist_list.append(avg_min_dist(estimated_layer_with_class, current_layer_with_class))
    prot_loss = torch.mean(torch.stack(mean_dist_list))
    output.append("Prototype Layer Loss：" + str(prot_loss.item()))
    output.append("%-----------------------------%")
    gnnNet.train()
    return prot_loss


def prot_sep_loss(estimated_prototype_layer, gnnNet):
    gnnNet.eval()
    prototype_layer = gnnNet.prototype_layer
    c, k, n = estimated_prototype_layer.shape

    min_distances = torch.zeros(c, dtype=torch.float32)

    for i in range(c):
        current_class_prototypes = estimated_prototype_layer[i, :, :]

        opposite_class_prototypes = torch.cat([prototype_layer[j, :, :] for j in range(c) if j != i], dim=0)

        distances = torch.norm(current_class_prototypes[:, None, :] - opposite_class_prototypes, dim=-1)
        dist = distances.min(dim=1).values
        min_distances[i] = dist.mean()

    average_min_distance = min_distances.mean()
    gnnNet.train()
    output.append("Prototype Layer Sep Loss：" + str(average_min_distance.item()))
    output.append("%-----------------------------%")
    return average_min_distance


def show_metrics(true_labels, predicted_labels, predicted_prob):
    f1 = f1_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    print(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    output.append(f"true F1 Score: {f1}, true Recall: {recall}, true Precision: {precision}")
    false_f1 = f1_score(true_labels, predicted_labels, pos_label=0)
    false_recall = recall_score(true_labels, predicted_labels, pos_label=0)
    false_precision = precision_score(true_labels, predicted_labels, pos_label=0)
    print(f"false F1 Score: {false_f1}, false Recall: {false_recall}, false Precision: {false_precision}")
    output.append(f"false F1 Score: {false_f1}, false Recall: {false_recall}, false Precision: {false_precision}")
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    micro_precision = precision_score(true_labels, predicted_labels, average='micro')
    micro_auc = roc_auc_score(true_labels, predicted_prob, average='micro')
    print(
        f"micro F1 Score: {micro_f1}, micro Recall: {micro_recall}, micro Precision: {micro_precision}, micro AUC Score: {micro_auc}")
    output.append(
        f"micro F1 Score: {micro_f1}, micro Recall: {micro_recall}, micro Precision: {micro_precision}, micro AUC Score: {micro_auc}")
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    macro_auc = roc_auc_score(true_labels, predicted_prob, average='macro')
    print(
        f"macro F1 Score: {macro_f1}, macro Recall: {macro_recall}, macro Precision: {macro_precision}, macro AUC Score: {macro_auc}")
    output.append(
        f"macro F1 Score: {macro_f1}, macro Recall: {macro_recall}, macro Precision: {macro_precision}, macro AUC Score: {macro_auc}")
    auc_value = roc_auc_score(true_labels, predicted_prob)
    print(f"AUC Score: {auc_value}")
    output.append(f"AUC Score: {auc_value}")
    """compute AUPRC"""
    p, r, t = precision_recall_curve(true_labels, predicted_prob)
    auprc = auc(r, p)
    print(f"AUPRC Score: {auprc}")
    output.append(f"AUPRC Score: {auprc}")
    print(
        f"{f1},{recall},{precision},{auprc},{auc_value},{false_f1},{false_recall},{false_precision},{micro_f1},{micro_recall},{micro_precision},{macro_f1},{macro_recall},{macro_precision}")
    output.append(
        f"{f1},{recall},{precision},{auprc},{auc_value},{false_f1},{false_recall},{false_precision},{micro_f1},{micro_recall},{micro_precision},{macro_f1},{macro_recall},{macro_precision}")
    return f1, recall, precision, auc_value, auprc


def train(gnnNet, train_dataloader, estimated_prot_layer):
    # 训练模式
    gnnNet.train()
    total_loss = 0.0
    total_prot_loss = 0.0
    total_crstent_loss = 0.0
    with tqdm(total=len(train_dataloader)) as pbar:
        for batch_data in train_dataloader:
            batch_data = batch_data.to(device)
            y_pred_score, labels = gnnNet(batch_data, "train", over_sample_scale_factor, sample_method, output)
            """loss = cross entropy loss + beta * prot_layer_loss + sep_beta * prot_sep_loss"""
            if estimated_prot_layer is None:
                loss = criterion(y_pred_score, labels)
                total_crstent_loss += loss.item()
            else:
                crst_ent = criterion(y_pred_score, labels)
                total_crstent_loss += crst_ent.item()
                original_p_loss = prototype_loss_with_mcts(gnnNet, estimated_prot_layer)
                original_p_sep_loss = prot_sep_loss(estimated_prot_layer, gnnNet)
                p_loss = original_p_loss * prot_beta_list[0]
                p_sep_loss = original_p_sep_loss * prot_beta_list[1]
                total_prot_loss += p_loss.item() - p_sep_loss.item()
                loss = crst_ent + p_loss - p_sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.update(1)
    average_loss = total_loss / len(train_dataloader)
    average_prot_loss = total_prot_loss / len(train_dataloader)
    average_crstent_loss = total_crstent_loss / len(train_dataloader)
    train_avg_loss.append((epoch, average_loss))
    train_prot_loss.append((epoch, average_prot_loss))
    train_crstent_loss.append((epoch, average_crstent_loss))
    print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {average_loss}, Avg Prototype Loss: {average_prot_loss}, Avg "
          f"Cross Entropy Loss: {average_crstent_loss}")
    output.append(f"Epoch {epoch}/{num_epochs}, Avg Loss: {average_loss}, Avg Prototype Loss: {average_prot_loss}, Avg "
                  f"Cross Entropy Loss: {average_crstent_loss}")


def validation(gnnNet, val_dataloader):
    gnnNet.eval()
    total_loss = 0.0
    val_true_labels = []
    val_predicted_prob = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            y_pred_score, labels = gnnNet(batch, "val", over_sample_scale_factor, sample_method, output)
            loss = criterion(y_pred_score, labels)
            total_loss += loss.item()
            val_predicted_prob.extend(y_pred_score.cpu().numpy())
            val_true_labels.extend(batch.y.cpu().numpy())

    val_predicted_labels = (np.array(val_predicted_prob) >= threshold).astype(int)

    show_metrics(val_true_labels, val_predicted_labels, val_predicted_prob)
    avg_loss = total_loss / len(val_dataloader)
    val_loss.append((epoch, avg_loss))
    print("validation avg loss:", avg_loss)
    output.append("validation avg loss:" + str(avg_loss))


def test(gnnNet, test_dataloader):
    gnnNet.eval()
    test_true_labels = []
    test_predicted_prob = []
    start_time = time.time()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            y_pred_score, labels = gnnNet(batch, "test", over_sample_scale_factor, sample_method, output)
            test_predicted_prob.extend(y_pred_score.cpu().numpy())
            test_true_labels.extend(batch.y.cpu().numpy())
    end_time = time.time()
    test_time = (end_time - start_time) / 3600

    test_predicted_labels = (np.array(test_predicted_prob) >= threshold).astype(int)

    f1, recall, precision, auc, auprc = show_metrics(test_true_labels, test_predicted_labels, test_predicted_prob)
    record_test_f1.append(f1)
    record_test_auprc.append(auprc)
    return test_time


def set_random_seed(seed=42):
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_best_threshold(y_true, y_pred_score):
    thresholds = np.arange(0.1, 1.0, 0.1)
    f1_thres_list = []
    # 计算精确度、召回率和F1分数
    for thres in thresholds:
        predicted_labels = (np.array(y_pred_score) >= thres).astype(int)
        f1 = f1_score(y_true, predicted_labels)
        rec = recall_score(y_true, predicted_labels)
        prec = precision_score(y_true, predicted_labels)
        f1_thres_list.append((f1, rec, prec, thres))
    f1_thres_list.sort(key=lambda x: x[0], reverse=True)
    print("f1 threshold:", f1_thres_list)
    output.append("f1 threshold: " + str(f1_thres_list))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # Argument and global variables
    parser = argparse.ArgumentParser('Interface for SLAD experiments.')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--node_feat_dim', type=int, default=484, help='feature dimension of dataset')
    parser.add_argument('--gnn_hidden_layer_size', type=int, help='the dimension of gnn layers',
                        default=64)
    parser.add_argument('--gnn_hidden_layer_no', type=int, help='the number of gnn layers',
                        default=6)
    parser.add_argument('--representation_num', type=int, help='the number of substructure representations', default=50)
    parser.add_argument('--dataset', type=str, default='halo', help='dataset name: forum/halo')

    args = parser.parse_args()

    dataset_name = "halo"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_of_class = 2
    repre_num_of_each_class = args.representation_num
    gnn_type = 'gtc'
    gnn_head_num = 1
    gnn_dropout = 0.0
    gnn_input_dim = args.node_feat_dim  # forum 247 halo 484
    prot_beta_list = [0.25, 0.0025]
    gnn_hidden_dim = [args.gnn_hidden_layer_size] * args.nn_hidden_layer_no
    mlp_input_dim = num_of_class * repre_num_of_each_class
    mlp_hidden_dim = 32
    mlp_output_dim = 1
    train_set_ratio = 0.8
    val_set_ratio = 0.1
    most_n_same_graph_structure = 0
    k = len(gnn_hidden_dim)
    use_specific_dataset = False
    dataloader_shuffle = True
    batch_size = args.bs
    train_dataloader, val_dataloader, test_dataloader, test_dataloader_unique, train_prot_dataloader, train_k_hop_graphs = load_dataset(
        dataset_name, 0, use_specific_dataset, most_n_same_graph_structure, train_set_ratio,
        val_set_ratio, dataloader_shuffle, batch_size, k)
    for pruning_strategy in ["soft_pruning"]:
        output = []  # results
        lr = 1e-3
        threshold = 0.5
        over_sample_scale_factor = 6
        sample_method = "SMOTE"  # SMOTE/copy
        activation_fuc = 'gelu'  # relu/leaky_relu/gelu
        num_epochs = args.n_epoch
        mcts_prot_epoch = [15, 18]
        mcts_top_bound = 100
        mcts_prot_selection = "kmeans"  # topk/kmeans
        mcts_pruning_strategy = pruning_strategy  # random/original/normal/soft_pruning
        random_negative_sample_mcts = True
        rollout_num = 2
        mcts_stop_strategy = "specific_size"
        min_atoms = 4
        k_neighbors = 200
        gnnNet = SLADGNN(gnn_head_num, gnn_dropout, activation_fuc, gnn_type, gnn_input_dim, gnn_hidden_dim,
                         num_of_class,
                         repre_num_of_each_class,
                         mlp_input_dim,
                         mlp_hidden_dim, mlp_output_dim).to(device)

        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(params=gnnNet.parameters(), lr=lr)
        "%----------start training------------%"
        estimated_prot_layer = None
        train_avg_loss = []
        train_crstent_loss = []
        train_prot_loss = []
        val_loss = []
        record_test_f1 = []
        record_test_auprc = []
        epoch = 1
        train_start_time = time.time()
        while epoch <= num_epochs:

            output.append(f"Epoch {epoch}/{num_epochs}, prototype:")
            for new_p_l in gnnNet.prototype_layer.tolist():
                output.append("%-----------------------------%")
                for new_p in new_p_l:
                    output.append(str(new_p))

            if epoch in mcts_prot_epoch:
                estimated_prot_layer = compute_estimated_prototype_layer(train_prot_dataloader, rollout_num,
                                                                         min_atoms,
                                                                         train_k_hop_graphs,
                                                                         gnnNet, repre_num_of_each_class,
                                                                         random_negative_sample_mcts,
                                                                         device, mcts_pruning_strategy,
                                                                         mcts_stop_strategy, k_neighbors,
                                                                         mcts_top_bound, mcts_prot_selection)

            train(gnnNet, train_dataloader, estimated_prot_layer)
            print("train avg loss history:\n", train_avg_loss)
            output.append("train avg loss history:\n" + str(train_avg_loss))
            print("train cross entropy loss history:\n", train_crstent_loss)
            output.append("train cross entropy loss history:\n" + str(train_crstent_loss))
            print("train prototype loss history:\n", train_prot_loss)
            output.append("train prototype loss history:\n" + str(train_prot_loss))
            print("%-----------------------------%")
            output.append("%-----------------------------%")
            epoch += 1

        "%---compute training time----%"
        train_end_time = time.time()
        train_time = (train_end_time - train_start_time) / 3600
        print("%-------train time(hrs)------%")
        output.append("%-------train time(hrs)------%")
        print("training time: " + str(train_time) + " hours, rollout number: " + str(rollout_num))
        output.append("training time: " + str(train_time) + " hours, rollout number: " + str(rollout_num))
        print("%-----------------------------%")
        output.append("%-----------------------------%")

        test_time = test(gnnNet, test_dataloader)
        output.append("%-------TEST COST(hrs)-------%")
        output.append(str(test_time))
        print("%-----------------------------%")
        output.append("%-----------------------------%")
        """%--------------WRITE RESULTS-----------------%"""
        with open(
                os.getcwd() + f"/slad_log.txt",
                'w',
                encoding='utf8') as write:
            for line in output:
                write.write(line + "\n")
