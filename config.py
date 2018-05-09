# -*- coding: utf-8 -*-

"""
@Author: oisc <oisc@outlook.com>
@Date: 2018/5/3
@Description: 配置文件
"""

import configparser
import os


__prefix = os.path.dirname(__file__)
_config = None
_config_file = os.path.join(__prefix, "config.ini")


def get(section, key=None, defult=..., rtype=str):
    global _config
    global _config_file
    if _config is None:
        _config = configparser.ConfigParser()
        _config.read(_config_file, encoding="utf-8")
    try:
        if key is None:
            return _config[section]
        else:
            if rtype is bool:
                return _config[section][key] == "true"
            else:
                return rtype(_config[section][key])
    except KeyError as e:
        if defult is ...:
            raise e
        else:
            return defult
