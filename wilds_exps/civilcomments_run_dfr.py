import argparse
import torch
import numpy as np
import tqdm
import pickle
import copy
from types import SimpleNamespace
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
from wilds_configs import datasets as dataset_configs
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds_models.initializer import initialize_model
from wilds_algorithms.initializer import infer_d_out

import transforms
from utils import parse_bool, save_pred

import sys
sys.path.append('../')
from dfr import dfr_run, dfr_tune_and_run, dfr_predict
# from models.text_models import _bert_replace_fc  #bert_pretrained


C_OPTIONS = [1., 0.3, 0.1, 0.07, 0.03, 0.01, 0.003]
THRESHOLD_OPTIONS = [-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15]
REG = "l1"
DATASET = 'civilcomments'


def get_data(args, wilds_config):
    full_dataset = get_dataset(dataset=DATASET, download=False, root_dir=args.root_dir)
    transform = transforms.initialize_transform(
            transform_name=wilds_config.transform,
            config=wilds_config,
            dataset=full_dataset,
            additional_transform_name=None,
            is_training=False)

    test_data = full_dataset.get_subset("test", transform=transform)
    val_data = full_dataset.get_subset("val", transform=transform)

    if args.dfr_reweighting_drop:
        train_data = full_dataset.get_subset("train", transform=transform)
        idx = train_data.indices.copy()
        rng = np.random.default_rng(args.dfr_reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - args.dfr_reweighting_frac) * len(idx))
        val_idx = idx[n_train:]
        reweighting_data = WILDSSubset(
                full_dataset,
                indices=val_idx,
                transform=transform)
        del train_data
    elif args.dfr_train:
        reweighting_data = full_dataset.get_subset("train", transform=transform)
    else:
        reweighting_data = val_data
    return val_data, test_data, reweighting_data


def load_model(args, wilds_config, d_out):
    model = initialize_model(config=wilds_config, d_out=d_out)
    ckpt_dict = torch.load(args.ckpt_path)
    if args.dfr_bert_model:
        model.load_state_dict({k.replace('fc.', 'classifier.'): v for (k, v) in ckpt_dict.items()})
    else:
        model.load_state_dict({k[len('model.'):]: v for (k, v) in ckpt_dict['algorithm'].items()})
    model.cuda()
    model.eval()
    classifier = model.classifier
    model.classifier = torch.nn.Identity(model.classifier.in_features)
    return model, classifier


def get_embeddings_predictions(feature_extractor, classifier, loader):
    all_embeddings, all_predictions, all_y_true, all_metadata = [], [], [], []
    # i = 0
    with torch.no_grad():
        for x, y_true, metadata in tqdm.tqdm(loader):
            embeddings = feature_extractor(x.cuda())
            predictions = torch.argmax(classifier(embeddings), axis=1)
            all_embeddings.append(embeddings.cpu())
            all_predictions.append(predictions.cpu())
            all_y_true.append(y_true.cpu())
            all_metadata.append(metadata)
            # i += 1
            # if i > 20:
            #     break
    all_embeddings = torch.cat(all_embeddings, axis=0)
    all_predictions = torch.cat(all_predictions, axis=0)
    all_y_true = torch.cat(all_y_true, axis=0)
    all_metadata = torch.cat(all_metadata, axis=0)
    return all_embeddings, all_predictions, all_y_true, all_metadata


def get_groups(args, metadata):
    if args.civilcomments_black:
        y = metadata[:, -2]
        black = metadata[:, 6]
        groups = y * 2 + black
    else:
        y = metadata[:, -2]
        identity_counts = metadata[:, :8].sum(axis=0)
        identity_ordering = torch.argsort(identity_counts)
        metadata_ordered = metadata[:, :8].T[identity_ordering].T
        groups = torch.argmax(torch.cat(
                [torch.zeros((len(metadata_ordered),))[:, None], metadata_ordered], axis=1), axis=1)
        groups = groups * 2 + y
    return groups


def dfr_tune_on_val(
        eval_fn, reweighting_embeddings, reweighting_y_true, reweighting_groups, 
        val_embeddings, val_y_true, val_metadata,
        verbose=False, preprocess=True
):
    all_results = {}
    best_wga = 0
    for c in C_OPTIONS:
        logreg, scaler = dfr_run(
                reweighting_embeddings, reweighting_y_true, reweighting_groups,
                num_retrains=10, preprocess=preprocess, reg=REG, verbose=verbose, c=c)
        for threshold in THRESHOLD_OPTIONS:
            shifted_logreg = copy.deepcopy(logreg)
            shifted_logreg.intercept_ += threshold
            dfr_val_predictions = torch.from_numpy(dfr_predict(val_embeddings, shifted_logreg, scaler))
            dfr_val_results = eval_fn(dfr_val_predictions, val_y_true, val_metadata)
            wga = dfr_val_results[0]['acc_wg']
            all_results[c, threshold] = wga
            if wga > best_wga:
                best_logreg = copy.deepcopy(shifted_logreg)
                best_wga = wga
                if verbose:
                    print('new best wga:', best_wga)
            if verbose:
                print(f'c={c}, t={threshold}: {wga}')
    if verbose:
        print(all_results)
    return best_logreg, scaler



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dfr_reweighting_drop', default=False, type=parse_bool, const=True, nargs='?',
                        help='Remove a dedicated reweighting data split from train data.')
    parser.add_argument('--dfr_reweighting_seed', default=0, type=int,
                        help='Random seed used for the reweighting data split.')
    parser.add_argument('--dfr_reweighting_frac', type=float, default=0.2,
                        help='Fraction of data to remove as a dedicated DFR reweighting data split.')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dfr_bert_model', default=False, type=parse_bool, const=True, nargs='?',
                        help='Use the BERT model from the DFR framework.')
    parser.add_argument('--prediction_path', type=str, default=None)
    parser.add_argument('--civilcomments_black', default=False, type=parse_bool, const=True, nargs='?',
                        help='Use only the black attribute to define groups.')
    parser.add_argument('--tune_on_val', default=False, type=parse_bool, const=True, nargs='?',
                        help='Tune on the original validation set and not reweighting dataset.')
    parser.add_argument('--dfr_train', default=False, type=parse_bool, const=True, nargs='?',
                        help='Retrain last layer on train data.')
    parser.add_argument('--dfr_tune_threshold', default=False, type=parse_bool, const=True, nargs='?',
                        help='Tune threshold for DFR logistic regression.')
    args = parser.parse_args()

    wilds_config = SimpleNamespace(
            algorithm='ERM',
            load_featurizer_only=False,
            pretrained_model_path=None,
            **dataset_configs.dataset_defaults[DATASET],
            )
    if args.dfr_bert_model:
        wilds_config.model = 'bert-base-uncased'
    wilds_config.model_kwargs = {}

    val_data, test_data, reweighting_data = get_data(args, wilds_config)
    d_out = infer_d_out(val_data, wilds_config)
    reweighting_loader = get_eval_loader("standard", reweighting_data, batch_size=args.batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)

    feature_extractor, classifier = load_model(args, wilds_config, d_out)

    reweighting_embeddings, _, reweighting_y_true, reweighting_metadata = get_embeddings_predictions(
            feature_extractor, classifier, reweighting_loader)
    val_embeddings, val_predictions, val_y_true, val_metadata = get_embeddings_predictions(
            feature_extractor, classifier, val_loader)
    test_embeddings, test_predictions, test_y_true, test_metadata = get_embeddings_predictions(
            feature_extractor, classifier, test_loader)
    base_model_val_results = test_data.eval(val_predictions, val_y_true, val_metadata)
    base_model_test_results = test_data.eval(test_predictions, test_y_true, test_metadata)
    print('Base model val:')
    print(base_model_val_results[-1])
    print('Base model test:')
    print(base_model_test_results[-1], end='\n\n')

    # DFR
    reweighting_groups = get_groups(args, reweighting_metadata)
    threshold_options = THRESHOLD_OPTIONS if args.dfr_tune_threshold else [0.]
    if args.tune_on_val:
        eval_fn = test_data.eval
        logreg, scaler = dfr_tune_on_val(
                eval_fn, reweighting_embeddings, reweighting_y_true, reweighting_groups, 
                val_embeddings, val_y_true, val_metadata,
                verbose=True, preprocess=True)
    else:
        logreg, scaler = dfr_tune_and_run(reweighting_embeddings, reweighting_y_true, reweighting_groups, verbose=True,
                                          threshold_options=threshold_options)
    dfr_val_predictions = torch.from_numpy(dfr_predict(val_embeddings, logreg, scaler))
    dfr_test_predictions = torch.from_numpy(dfr_predict(test_embeddings, logreg, scaler))
    dfr_val_results = test_data.eval(dfr_val_predictions, val_y_true, val_metadata)
    dfr_test_results = test_data.eval(dfr_test_predictions, test_y_true, test_metadata)
    print('DFR val:')
    print(dfr_val_results[-1])
    print('DFR test:')
    print(dfr_test_results[-1])

    # Save
    if args.prediction_path:
        save_pred(dfr_test_predictions, args.prediction_path + 'test')
        save_pred(dfr_val_predictions, args.prediction_path + 'val')
        with open(args.prediction_path + '_results.pkl', 'wb') as f:
            pickle.dump({
                    'base_model_val': base_model_val_results,
                    'base_model_test': base_model_test_results,
                    'dfr_val': dfr_val_results,
                    'dfr_test': dfr_test_results,
            }, f)


if __name__=='__main__':
    main()
