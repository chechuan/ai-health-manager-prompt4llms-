from urllib import parse
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.exc import OperationalError

try:
    from src.utils.Logger import logger
except Exception as err:
    from Logger import logger


class MysqlConnector:
    def __init__(
        self,
        user: str = None,
        passwd: str = None,
        ip: str = "0.0.0.0",
        port: int = 3306,
        db_name: str = "localhost",
    ) -> None:
        passwd = parse.quote_plus(passwd)
        self.url = f"mysql+pymysql://{user}:{passwd}@{ip}:{port}/{db_name}"
        self.engine = create_engine(self.url)
        self.metadata = MetaData(self.engine)
        self.is_connected = False  # 记录数据库连接状态

        try:
            self.connect = self.engine.connect()
            self.is_connected = True
            logger.info("MySQL 连接成功")
        except OperationalError as err:
            logger.error(f"MySQL 连接失败: {err}")
            self.is_connected = False  # 连接失败，设置标志位

    def reconnect(self):
        """尝试重连 MySQL"""
        try:
            self.engine = create_engine(self.url)
            self.metadata = MetaData(self.engine)
            self.connect = self.engine.connect()
            self.is_connected = True
            logger.info("MySQL 重连成功")
        except OperationalError as err:
            logger.error(f"MySQL 重连失败: {err}")
            self.is_connected = False  # 连接失败，标志为 False，不再尝试执行 SQL

    def insert(self, table_name, datas):
        """数据库插入"""
        if not self.is_connected:
            logger.error("数据库未连接，插入操作跳过")
            return

        table_obj = Table(table_name, self.metadata, autoload=True)
        try:
            self.connect.execute(table_obj.insert(), datas)
        except OperationalError as error:
            logger.error(f"插入失败: {error}")
            self.reconnect()
            if self.is_connected:
                try:
                    self.connect.execute(table_obj.insert(), datas)
                except Exception as err:
                    logger.error(f"重连后仍然插入失败: {err}")

        print(table_name, "\tinsert ->\t", len(datas))

    def query(self, sql, orient="records"):
        """数据库查询"""
        if not self.is_connected:
            logger.error("数据库未连接，查询操作跳过")
            return {}

        try:
            return pd.read_sql_query(sql, self.connect).to_dict(orient=orient)
        except OperationalError as error:
            logger.error(f"查询失败: {error}")
            self.reconnect()
            if self.is_connected:
                try:
                    return pd.read_sql_query(sql, self.connect).to_dict(orient=orient)
                except Exception as err:
                    logger.error(f"重连后仍然查询失败: {err}")

        return {}

    def execute(self, sql):
        """执行 SQL 语句"""
        if not self.is_connected:
            logger.error("数据库未连接，SQL 执行跳过")
            return

        try:
            return self.connect.execute(sql)
        except OperationalError as error:
            logger.error(f"SQL 执行失败: {error}")
            self.reconnect()
            if self.is_connected:
                try:
                    return self.connect.execute(sql)
                except Exception as err:
                    logger.error(f"重连后仍然执行失败: {err}")