import mysql.connector
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 连接到MySQL数据库
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host="192.168.31.120",
            user="root",
            password="856262",  # 替换为你的MySQL密码
            database="text2sql"
        )
        return conn
    except Exception as e:
        print(f"数据库连接错误: {e}")
        return None

# 加载预训练模型
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        return tokenizer, model
    except Exception as e:
        print(f"模型加载错误: {e}")
        return None, None

# 简单的规则匹配方法
def rule_based_text2sql(query):
    # 定义表结构信息
    table_info = {
        "表名": "employees",
        "字段": ["id", "name", "age", "gender", "department", "position", "salary", "hire_date", "email", "phone"]
    }
    
    # 字段中文映射
    field_mapping = {
        "姓名": "name",
        "年龄": "age",
        "性别": "gender",
        "部门": "department",
        "职位": "position",
        "薪资": "salary",
        "入职日期": "hire_date",
        "邮箱": "email",
        "电话": "phone"
    }
    
    # 查询类型识别
    if re.search(r"查询|显示|列出|获取", query):
        # 构建SELECT语句
        sql = "SELECT * FROM employees"
        
        # 条件匹配
        conditions = []
        
        # 部门匹配
        dept_match = re.search(r"(技术部|市场部|人力资源|财务部)", query)
        if dept_match:
            conditions.append(f"department = '{dept_match.group(1)}'")
        
        # 性别匹配
        gender_match = re.search(r"(男|女)", query)
        if gender_match:
            conditions.append(f"gender = '{gender_match.group(1)}'")
        
        # 年龄匹配
        age_match = re.search(r"(\d+)岁", query)
        if age_match:
            conditions.append(f"age = {age_match.group(1)}")
        
        # 薪资范围匹配
        salary_match = re.search(r"薪资(大于|高于|超过)(\d+)", query)
        if salary_match:
            conditions.append(f"salary > {salary_match.group(2)}")
        
        salary_match = re.search(r"薪资(小于|低于)(\d+)", query)
        if salary_match:
            conditions.append(f"salary < {salary_match.group(2)}")
        
        # 添加WHERE子句
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        # 排序匹配
        if re.search(r"按(薪资|年龄)(升序|降序|从高到低|从低到高)", query):
            order_field = "salary" if "薪资" in query else "age"
            order_type = "DESC" if re.search(r"降序|从高到低", query) else "ASC"
            sql += f" ORDER BY {order_field} {order_type}"
        
        # 限制结果数量
        limit_match = re.search(r"前(\d+)名|限制(\d+)条|最多(\d+)条", query)
        if limit_match:
            limit = limit_match.group(1) or limit_match.group(2) or limit_match.group(3)
            sql += f" LIMIT {limit}"
        
        return sql
    
    # 统计查询
    elif re.search(r"统计|计算|平均|总数|人数", query):
        if re.search(r"平均(薪资|年龄)", query):
            field = "salary" if "薪资" in query else "age"
            sql = f"SELECT AVG({field}) FROM employees"
        elif re.search(r"(总数|人数)", query):
            sql = "SELECT COUNT(*) FROM employees"
        else:
            sql = "SELECT COUNT(*) FROM employees"
        
        # 条件匹配
        conditions = []
        
        # 部门匹配
        dept_match = re.search(r"(技术部|市场部|人力资源|财务部)", query)
        if dept_match:
            conditions.append(f"department = '{dept_match.group(1)}'")
        
        # 性别匹配
        gender_match = re.search(r"(男|女)", query)
        if gender_match:
            conditions.append(f"gender = '{gender_match.group(1)}'")
        
        # 添加WHERE子句
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        return sql
    
    else:
        return "SELECT * FROM employees LIMIT 10"  # 默认查询

# 执行SQL并返回结果
def execute_query(conn, sql):
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        print(f"查询执行错误: {e}")
        return []

# 主函数
def main():
    conn = connect_to_db()
    if not conn:
        return
    
    print("欢迎使用Text2SQL人员查询系统！")
    print("示例查询:")
    print("1. 查询技术部的所有员工")
    print("2. 显示薪资大于15000的员工")
    print("3. 统计技术部的平均薪资")
    print("4. 列出按薪资降序排列的前5名员工")
    print("5. 输入'退出'结束程序")
    
    while True:
        query = input("\n请输入您的查询 (自然语言): ")
        if query.lower() in ['退出', 'exit', 'quit']:
            break
        
        # 将自然语言转换为SQL
        sql = rule_based_text2sql(query)
        print(f"\n生成的SQL: {sql}")
        
        # 执行SQL并显示结果
        results = execute_query(conn, sql)
        
        if results:
            # 打印表头
            headers = results[0].keys()
            header_str = " | ".join(headers)
            print("\n" + "-" * len(header_str))
            print(header_str)
            print("-" * len(header_str))
            
            # 打印数据行
            for row in results:
                row_values = [str(value) for value in row.values()]
                print(" | ".join(row_values))
            
            print("-" * len(header_str))
            print(f"共 {len(results)} 条记录")
        else:
            print("没有找到匹配的记录")
    
    conn.close()
    print("感谢使用，再见！")

if __name__ == "__main__":
    main()