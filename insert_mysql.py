import pandas as pd
import mysql.connector
import logging

# 数据库连接配置
db_config = {
    'host': '10.39.10.28',
    'port': 3306,
    'user': 'root',
    'password': 'fpBase09@23aQdn',
    'database': 'ai_health_manager_db'
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_table(cursor, table_name, df):
    # 动态创建表，字段根据DataFrame列名生成
    columns = ", ".join([f"{col} TEXT" for col in df.columns])
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
    cursor.execute(create_table_query)

def clean_nan_values(df):
    # 将DataFrame中的NaN替换为None，以便插入MySQL时转换为NULL
    return df.where(pd.notna(df), None)

def import_data_to_mysql(file_path, table_name, cursor, conn):
    # 读取Excel数据
    df = pd.read_excel(file_path)

    # 清洗NaN值
    df = clean_nan_values(df)

    # 创建表
    create_table(cursor, table_name, df)
    logging.info(f"表 {table_name} 创建成功或已存在")

    # 插入数据
    for index, row in df.iterrows():
        try:
            columns = ", ".join(row.index)
            placeholders = ", ".join(["%s"] * len(row))
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(insert_query, tuple(row))
        except Exception as e:
            logging.error(f"插入第 {index + 1} 行数据失败: {e}")

    # 提交事务
    conn.commit()
    logging.info(f"数据成功导入到表: {table_name}")

def main():
    # 建立数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # 处理两个Excel文件
        for file_name in ["cleaned_packages.xlsx", "cleaned_activities.xlsx"]:
            file_path = f'doc/bath_plan/{file_name}'
            table_name = file_name.split('.')[0]  # 根据文件名生成表名
            logging.info(f"正在处理文件: {file_path}")
            import_data_to_mysql(file_path, table_name, cursor, conn)

    except Exception as e:
        logging.error(f"出现错误: {e}")
        conn.rollback()

    finally:
        # 关闭数据库连接
        cursor.close()
        conn.close()
        logging.info("数据库连接已关闭")

if __name__ == "__main__":
    main()
