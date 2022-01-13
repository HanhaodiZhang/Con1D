def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch, lr_adj, n):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # total_epochs = args.num_epochs
    if args.base_lr * lr_adj <5e-8:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-8
        return 5e-8, lr_adj
    warm_epochs = args.warmup_epochs
    if epoch < warm_epochs:
        lr_adj *= (batch_iter + 1) / (warm_epochs * train_batch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.base_lr * lr_adj
        return args.base_lr * lr_adj, 1
    elif epoch % args.reduce_n_epoch == 0 and n == 0:
        lr_adj *= args.reduce_rate
    else:
        lr_adj = lr_adj

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * lr_adj
    return args.base_lr * lr_adj, lr_adj
