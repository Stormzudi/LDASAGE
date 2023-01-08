# -*- coding: utf-8 -*-
import os
from datetime import datetime
from config import DIRPATH, log_save_days
from loguru import logger
import re
import time


class Loging:
    __instance = None
    date = datetime.now().strftime('%Y-%m-%d')
    logpath = os.path.join(DIRPATH, "log")
    if not os.path.isdir(logpath):
        os.makedirs(logpath)

    logger.add('%s/%s.log' % (logpath, date),
               format="{time:YYYY-MM-DD HH:mm:ss}  | {level}> {elapsed}  | {message}",
               encoding='utf-8',
               retention='7 days',
               backtrace=True,
               diagnose=True,
               enqueue=True
               )

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Loging, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def info(self, msg, *args, **kwargs):
        return logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        return logger.exception(msg, *args, exc_info=True, **kwargs)

    def clear_exprie_logs(self, *args, **kwargs):
        logpath = os.path.join(DIRPATH, "log/")
        log_list_src = list(os.listdir(logpath))
        curr_time = time.time()
        for ind in range(len(log_list_src)):
            match = re.search('\d+-\d+-\d+', log_list_src[ind])
            if not match:
                continue
            log_date = match.group()
            log_date_dt = datetime.strptime(log_date, '%Y-%m-%d').timestamp()
            time_diff = (curr_time - log_date_dt) / 86400

            # 超过设定的 log时间（log_save_days=15）将 全部删除文件
            if time_diff >= log_save_days:
                try:
                    os.remove(logpath + log_list_src[ind])
                    logger.info("Clear expried log：%s" % log_list_src[ind])
                except Exception as e:
                    logger.error(e)


logs = Loging()
logs.clear_exprie_logs()
