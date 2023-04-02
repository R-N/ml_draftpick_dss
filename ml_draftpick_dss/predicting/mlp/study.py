import torch
from .dataset import create_dataloader, ResultDataset
from .model import GlobalPooling1D
from .predictor import ResultPredictor
from ..embedding import scaled_sqrt_factory, create_embedding_sizes

def study(
    datasets,
    encoder,
    id=1,
    d_input = 171,
    d_final = 128,
    d_hid_encoder = 128,
    n_layers_encoder = 2,
    bias_encoder = True,
    d_hid_final = 128,
    n_layers_final = 3,
    bias_final = True,
    dropout = 0.1,
    lrs = [1e-3, 1e-4, 1e-5],
    epochs = [100, 100, 100],
    norm_crit = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam,
    grad_clipping = 0,
    batch_size = 128,
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    predictor = ResultPredictor(
        d_input,
        d_final,
        encoder_kwargs={
            "d_hid": d_hid_encoder,
            "n_layers": n_layers_encoder,
            "activation": torch.nn.ReLU,
            "bias": bias_encoder,
            "dropout": dropout,
        }, 
        final_kwargs={
            "d_hid": d_hid_final,
            "n_layers": n_layers_final,
            "activation": torch.nn.ReLU,
            "bias": bias_final,
            "dropout": dropout,
        },
        head_kwargs={
            "heads": ["victory", "score", "duration"],
            "bias": bias_final,
            "dropout": dropout,
        },
    )
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lrs[0],
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
    )
    for lr, epoch in zip(lrs, epochs):
        predictor.set_lr(lr)
        for i in range(epoch):
            print(predictor.train())
            print(predictor.train(val=True))
    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    return best_metrics["val_loss"]