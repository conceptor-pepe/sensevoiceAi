#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SenseVoice API 应用入口
启动API服务
"""

import os
import uvicorn
import config
from api import app

def main():
    """
    应用主入口
    """
    # 打印服务配置信息
    config.print_config()
    
    # 启动服务
    uvicorn.run(
        app, 
        host=config.API_HOST, 
        port=config.API_PORT
    )

if __name__ == "__main__":
    main() 