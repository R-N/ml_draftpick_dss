import torch
from ..dataset import create_dataloader
from .model import GlobalPooling1D
from .predictor import ResultPredictor
from ..embedding import scaled_sqrt_factory, create_embedding_sizes

def study(
    dfs,
    encoder,
    id=1,
    n_heads = 2,
    d_hid_tf = 128,
    n_layers_tf = 2,
    d_hid_final = 128,
    n_layers_final = 3,
    bias_final = True,
    dropout = 0.1,
    pos_encoder = False,
    bidirectional = False,
    pooling = GlobalPooling1D(),
    lrs = [1e-3, 1e-4, 1e-5],
    epochs = [100, 100, 100],
    norm_crit = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam,
    grad_clipping = 0,
    batch_size = 128,
):
    train_df, val_df, test_df = dfs
    train_loader = create_dataloader(train_df, encoder, batch_size=batch_size)
    val_loader = create_dataloader(val_df, encoder, batch_size=batch_size)
    test_loader = create_dataloader(test_df, encoder, batch_size=batch_size)

    sizes = create_embedding_sizes(
        encoder.x.columns[1:], 
        f=scaled_sqrt_factory(n_heads)
    )
    predictor = ResultPredictor(
        sizes, 
        encoder_kwargs={
            "n_heads": n_heads,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
            "dropout": dropout,
        }, 
        decoder_kwargs={
            "n_heads": n_heads,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
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
        pos_encoder=pos_encoder,
        bidirectional=bidirectional,
        pooling=pooling,
    )
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lrs[0],
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
        checkpoint_dir=f"studies/{id}/checkpoints",
        log_dir=f"studies/{id}/logs",
    )
    for lr, epoch in zip(lrs, epochs):
        predictor.set_lr(lr)
        for i in range(epoch):
            print(predictor.train())
            print(predictor.train(val=True))
    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    return best_metrics["val_loss"]