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

    # INFO日志输出到info.log
    info_handler = RotatingFileHandler(
        os.path.join(log_dir, 'info.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    info_handler.setFormatter(info_formatter)


     # debug日志输出到debug.log
    debug_handler = RotatingFileHandler(
        os.path.join(log_dir, 'debug.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)  # 添加过滤器
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)

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
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug('调试信息')
    logger.info('普通信息')
    logger.warning('警告信息')
    logger.error('错误信息')
    logger.critical('严重错误')