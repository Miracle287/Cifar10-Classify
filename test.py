#!/usr/bin/python
# -*- coding: UTF-8 -*-
# description: 

__author__ = "Zhenyu Lee"
__copyright__ = "Copyright 2019, Fuzhou University"
__version__ = "1.0"
__status__ = "Development"

import inputs
import train


def show_history_from_file(file):
    history = inputs.load(file)
    train.show_history(history)


show_history_from_file(inputs.output_dir + "/train_history")


