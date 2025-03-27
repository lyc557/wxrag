import logging
import os
from logging.handlers import RotatingFileHandler

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

def get_logger(name):
    """获取配置好的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置根日志级别
    
    # 防止重复添加handler
    if logger.handlers:
        return logger

    # 所有日志输出到app.log
    all_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    all_handler.setLevel(logging.DEBUG)
    all_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    all_handler.setFormatter(all_formatter)

    # 错误日志输出到error.log
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.WARNING)
    error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加handler
    logger.addHandler(all_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger