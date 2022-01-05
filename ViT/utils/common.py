#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/06 10:27 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : common.py
# @Software  : PyCharm

import numpy

def uniform_ball(batch_size, dim, epsilon=1, ord=2, alternative_mode=True):
    """
    Sample vectors uniformly in the n-ball.
    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.
    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :param alternative_mode: whether to sample from uniform distance instead of sampling uniformly with respect to volume
    :type alternative_mode: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon
    if alternative_mode:
        uniform = numpy.random.uniform(0, 1, (batch_size, 1)) # exponent is only difference!
    else:
        uniform = numpy.random.uniform(0, 1, (batch_size, 1)) ** (1. / dim)
    random *= numpy.repeat(uniform, axis=1, repeats=dim)

    return random