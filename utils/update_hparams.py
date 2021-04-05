def update_hparams(
    hparams, dropout, batch_size, lr, n_epochs,
):

    if dropout is not None:
        hparams['model']['dropout_rate'] = float(dropout)

    if batch_size is not None:
        hparams['batch_size'] = int(batch_size)

    if lr is not None:
        hparams['optimizer_hparams']['lr'] = float(lr)

    if n_epochs is not None:
        hparams['n_epochs'] = int(n_epochs)

    return hparams
