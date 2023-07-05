import os
import time
import torch
import models
import optimizers
import losses
from losses import get_exp_weights
import utils
from utils.common_utils import get_embeddings_args
from utils.common_utils import set_seed
from utils.common_utils import get_embeddings_loader
from utils.supervised_utils import get_classifier_and_feature_extractor
from utils.supervised_utils import take_step
from utils.supervised_utils import get_last_layer_model
from utils.supervised_utils import rebalance_weights
from utils.supervised_utils import normalize_weights
from utils.supervised_utils import EmbeddingManager
from utils.logging import EmbeddingsLogger
from utils.general import print_time_taken


def train_embeddings(args):
    assert args.emb_batch_size == -1, f'AFR assumes full batch size (args.emb_batch_size = -1), but got {args.emb_batch_size}'
    tic = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    Log = EmbeddingsLogger(args)

    train_loader, holdout_loaders = utils.get_data(args, finetune_on_val=args.finetune_on_val)
    Log.log_data_stats(train_loader, holdout_loaders['test'], holdout_loaders['val'])

    resume_ckpt = os.path.join(args.base_model_dir, args.checkpoint)
    model = getattr(models, args.model)(holdout_loaders["val"].dataset.n_classes).to(device)
    aux = torch.load(resume_ckpt, map_location=device)
    if isinstance(aux, dict):
        model.load_state_dict(aux)
    else:
        model = aux
    model.eval()
    Log.logger.info(f'Model has {utils.count_parameters(model) / 1e6:.2g}M parameters')

    classifier, feature_extractor = get_classifier_and_feature_extractor(model)
    Embedder = EmbeddingManager(device, args)
    loaders = (train_loader, holdout_loaders)
    all_out = Embedder.get_train_test_embeddings(classifier, feature_extractor, loaders)
    out, out_test, out_val = all_out
    test_embeddings, _, test_groups, test_y = out_test
    val_embeddings, _, val_groups, val_y = out_val
    embeddings, _, groups, y = out
    logits = classifier(embeddings).detach()
    weights = get_exp_weights(logits, y, gamma=args.gamma)
    if args.rebalance_weights:
        weights = rebalance_weights(weights, y)
    weights = normalize_weights(weights)
    loader = get_embeddings_loader(args.emb_batch_size, embeddings, y, weights)
    init_weights = (classifier.weight.detach().clone(), classifier.bias.detach().clone())
    Log.log_group_weights(weights, groups, epoch=0)

    logits_test = classifier(test_embeddings)
    _ = Log.generate_acc_groups(logits_test, test_y, test_groups, epoch=0, partition="test")
    Log.save_holdout_data(test_data=(test_embeddings, test_y, test_groups),
                          val_data=(val_embeddings, val_y, val_groups))

    last_layer = get_last_layer_model(init_weights).to(device)
    last_layer.train()
    optimizer = getattr(optimizers, args.optimizer)(last_layer, args)
    scheduler = getattr(optimizers, args.scheduler)(optimizer, args)
    criterion = getattr(losses, args.loss)()

    for epoch in range(args.num_epochs):
        all_logits = []
        for ds in loader:
            loss, logits = take_step(last_layer, ds, optimizer, criterion, scheduler, init_weights,
                                     args.reg_coeff, max_norm=args.grad_norm)
            all_logits.append(logits)
        logits = torch.concat(all_logits, dim=0)
        if epoch > -1:
            Log.logger.info(f"\nE: {epoch} | L: {loss:1.5e} ")
            _ = Log.generate_acc_groups(logits, y, groups, epoch + 1, "train")
            holdout_logits = (last_layer(test_embeddings), last_layer(val_embeddings))
            Log.log_results_save_chkp(epoch + 1, holdout_logits, model=last_layer)

    Log.finalize_logging(last_layer)
    Log.save_plot("./logs/")
    toc = time.time()
    print_time_taken(toc - tic)


if __name__ == '__main__':
    parser = get_embeddings_args()
    args = parser.parse_args()
    train_embeddings(args)
