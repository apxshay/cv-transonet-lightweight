#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Learning rate policy."""

import math
solver_steps = []

def get_lr_at_epoch(cur_epoch): #cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func("cosine")(cur_epoch)
    # Perform warm up.
    if cur_epoch < 70.0:
        lr_start =  1e-8
        lr_end = get_lr_func("cosine")(70.0)
        alpha = (lr_end - lr_start) / 70.0
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = 70.0 #if cfg.SOLVER.COSINE_AFTER_WARMUP else 0.0
    #assert 1e-6 < 0.00025
    return (
        1e-6
        + (0.00025 - 1e-6)
        * (
            math.cos(math.pi * (cur_epoch - offset) / (300 - offset))
            + 1.0
        )
        * 0.5
    )


# def lr_func_steps_with_relative_lrs(cur_epoch):#cfg, cur_epoch):
#     """
#     Retrieve the learning rate to specified values at specified epoch with the
#     steps with relative learning rate schedule.
#     Args:
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#         cur_epoch (float): the number of epoch of the current training stage.
#     """
#     ind = get_step_index(cur_epoch)#cfg, cur_epoch)
#     return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cur_epoch):#cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """

    steps = solver_steps + [300]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]