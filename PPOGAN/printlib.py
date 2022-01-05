#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/16 10:35 오전
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : printlib.py
# @Software  : PyCharm

def print_(v="", progress="", caption="", value=""):
    string = "#####"
    if progress != "" or caption != "" or value != "":
        print(string + v + "$" + progress + "$" + caption + "$" + str(value) + string)
    else:
        return


def print_normal(v=None, progress=None, caption=None, value=None):
    if v is None:
        print(progress)
    print(v, progress)
