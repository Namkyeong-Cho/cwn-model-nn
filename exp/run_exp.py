import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import random

from data.data_loading import DataLoader, load_dataset
from torch_geometric.data import DataLoader as PyGDataLoader
from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser, validate_args
from mp.graph_models import GIN0, GINWithJK
from mp.models import CIN0, Dummy, SparseCIN, EdgeOrient, EdgeMPNN, MessagePassingAgnostic
from mp.molec_models import EmbedSparseCIN, OGBEmbedSparseCIN, EmbedSparseCINNoRings, EmbedGIN
from mp.ring_exp_models import RingSparseCIN, RingGIN


def main(args):
    """The common training and evaluation script used by all the experiments."""
    # set device
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print("==========================================================")
    print("Using device", str(device))
    print(f"Fold: {args.fold}")
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create results folder
    result_folder = os.path.join(
        args.result_folder, f'{args.dataset}-{args.exp_name}', f'seed-{args.seed}')
    if args.fold is not None:
        result_folder = os.path.join(result_folder, f'fold-{args.fold}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')
    dataset = load_dataset(args.dataset,
                           task=args.task,
                           node_feature_type= args.node_feature_type,
                           max_dim=args.max_dim, fold=args.fold,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           flow_points=args.flow_points, flow_classes=args.flow_classes,
                           max_ring_size=args.max_ring_size,
                           use_edge_features=args.use_edge_features,
                           simple_features=args.simple_features, n_jobs=args.preproc_jobs,
                           train_orient=args.train_orient, test_orient=args.test_orient)
    print("datset called!:", args.tune)
    print(dataset.get_split('train'), dataset.get_split('test'))
    # assert False

    for data in dataset:
        print("data type : ", type(data))
        print("data : ", data.model_nm)
    if args.tune:
        split_idx = dataset.get_tune_idx_split()
    else:
        split_idx = dataset.get_idx_split()

    # Instantiate data loaders
    train_loader = DataLoader(dataset.get_split('train'), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    # valid_loader = DataLoader(dataset.get_split('valid'), batch_size=args.batch_size,
    #     shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)
    # test_split = split_idx.get("test", None)
    # test_loader = None
    # if test_split is not None:
    test_loader = DataLoader(dataset.get_split('test'), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)

    # Automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric, eps=args.iso_eps)

    # # Use coboundaries?
    # use_coboundaries = args.use_coboundaries.lower() == 'true'
    #
    # # Readout dimensions
    # readout_dims = tuple(sorted(args.readout_dims))

    # Instantiate model
    if args.model == 'cin':
        print("CIN0 calling")
        print("dataset num feature :", dataset.num_features_in_dim(0))
        print("dataset num classes :",dataset.num_classes)
        print("dataset num layers :",args.num_layers)
        print("dataset num feature :",args.emb_dim)
        model = CIN0(dataset.num_features_in_dim(0),          # num_input_features
                     dataset.num_classes,                     # num_classes
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     dropout_rate=args.drop_rate,             # dropout rate
                     max_dim=dataset.max_dim,                 # max_dim
                     jump_mode=args.jump_mode,                # jump mode
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
                    ).to(device)

    # print("============= Model Parameters =================")
    # trainable_params = 0
    # total_params = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())
    #         trainable_params += param.numel()
    #     total_params += param.numel()
    # print("============= Params stats ==================")
    # print(f"Trainable params: {trainable_params}")
    # print(f"Total params    : {total_params}")

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # instantiate learning rate decay
    if args.lr_scheduler == 'ReduceLROnPlateau':
        mode = 'min' if args.minimize else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                               factor=args.lr_scheduler_decay_rate,
                                                               patience=args.lr_scheduler_patience,
                                                               verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []
    if not args.untrained:
        for epoch in range(1, args.epochs + 1):

            # perform one epoch
            print("=====Epoch {}".format(epoch))
            print('Training...')
            epoch_train_curve = train(model, device, train_loader, optimizer, args.task_type)
            train_loss_curve += epoch_train_curve
            epoch_train_loss = float(np.mean(epoch_train_curve))

            # evaluate model
            print('Evaluating...')
            if epoch == 1 or epoch % args.train_eval_period == 0:
                train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
            train_curve.append(train_perf)

            # valid_perf, epoch_val_loss = eval(model, device,
            #     valid_loader, evaluator, args.task_type)#, dataset[split_idx["valid"]])

            valid_perf, epoch_val_loss = eval(model, device, test_loader, evaluator,
                                              args.task_type)
            valid_curve.append(valid_perf)
            if test_loader is not None:
                test_perf, epoch_test_loss = eval(model, device, test_loader, evaluator,
                                                  args.task_type)

            else:
                test_perf = np.nan
                epoch_test_loss = np.nan
            test_curve.append(test_perf)

            print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
                  f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
                  f' | Test Loss {epoch_test_loss:.3f}')

            # decay learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_perf)
                    # We use a strict inequality here like in the benchmarking GNNs paper code
                    # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
                    if args.early_stop and optimizer.param_groups[0]['lr'] < args.lr_scheduler_min:
                        print("\n!! The minimum learning rate has been reached.")
                        break
                else:
                    scheduler.step()

            i = 0
            new_params = []
            if epoch % args.train_eval_period == 0:
                print("====== Slowly changing params ======= ")
            for name, param in model.named_parameters():
                # print(f"Param {name}: {param.data.view(-1)[0]}")
                # new_params.append(param.data.detach().clone().view(-1)[0])
                new_params.append(param.data.detach().mean().item())
                if len(params) > 0 and epoch % args.train_eval_period == 0:
                    if abs(params[i] - new_params[i]) < 1e-6:
                        print(f"Param {name}: {params[i] - new_params[i]}")
                i += 1
            params = copy.copy(new_params)

        if not args.minimize:
            best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
    else:
        train_loss_curve.append(np.nan)
        train_curve.append(np.nan)
        valid_curve.append(np.nan)
        test_curve.append(np.nan)

    print('Final Evaluation...')
    final_train_perf = np.nan
    final_val_perf = np.nan
    final_test_perf = np.nan
    print("args.dataset :" ,args.dataset)
    final_train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
    _, _, input_dict = eval(model, device, test_loader, evaluator, args.task_type,
                                           show_input_dict=True)
    print("inupt_dict : ", input_dict)
    for idx,test_data in enumerate(dataset.get_split('test')):
        print("complex :", test_data.model_nm)
        print("y : ", test_data.y)
        print("input_dict y_true: ",input_dict['y_true'][idx])

    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'last_train': final_train_perf,
        'best': best_val_epoch}

    msg = (
       f'========== Result ============\n'
       f'Dataset:        {args.dataset}\n'
       f'------------ Best epoch -----------\n'
       f'Train:          {train_curve[best_val_epoch]}\n'
       f'Validation:     {valid_curve[best_val_epoch]}\n'
       f'Test:           {test_curve[best_val_epoch]}\n'
       f'Best epoch:     {best_val_epoch}\n'
       '------------ Last epoch -----------\n'
       f'Train:          {final_train_perf}\n'
       f'Validation:     {final_val_perf}\n'
       f'Test:           {final_test_perf}\n'
       '-------------------------------\n\n')
    print(msg)

    msg += str(args)
    with open(filename, 'w') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(os.path.join(result_folder, 'curves.pkl'), 'wb') as handle:
            pickle.dump(curves, handle)

    return curves


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)

    main(args)
