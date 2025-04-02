-- 创建一个名为 "employees" 的表
CREATE TABLE employees (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '员工ID，自增主键',
    name VARCHAR(50) NOT NULL COMMENT '员工姓名',
    age INT COMMENT '员工年龄',
    gender VARCHAR(10) COMMENT '性别',
    department VARCHAR(50) COMMENT '所属部门',
    position VARCHAR(50) COMMENT '职位名称',
    salary DECIMAL(10, 2) COMMENT '月薪，精确到小数点后2位',
    hire_date DATE COMMENT '入职日期',
    email VARCHAR(100) COMMENT '电子邮箱地址',
    phone VARCHAR(20) COMMENT '联系电话'
);

-- 插入一些示例数据
INSERT INTO employees (name, age, gender, department, position, salary, hire_date, email, phone)
VALUES 
    ('张三', 28, '男', '技术部', '高级工程师', 15000.00, '2020-03-15', 'zhangsan@example.com', '13800138001'),
    ('李四', 35, '男', '市场部', '市场经理', 18000.00, '2018-05-20', 'lisi@example.com', '13800138002'),
    ('王五', 26, '女', '人力资源', 'HR专员', 9000.00, '2021-07-10', 'wangwu@example.com', '13800138003'),
    ('赵六', 40, '男', '财务部', '财务总监', 25000.00, '2015-11-05', 'zhaoliu@example.com', '13800138004'),
    ('钱七', 31, '女', '技术部', '产品经理', 16000.00, '2019-02-18', 'qianqi@example.com', '13800138005'),
    ('孙八', 29, '男', '技术部', '前端工程师', 14000.00, '2020-06-22', 'sunba@example.com', '13800138006'),
    ('周九', 33, '女', '市场部', '市场专员', 10000.00, '2019-09-15', 'zhoujiu@example.com', '13800138007'),
    ('吴十', 45, '男', '技术部', '技术总监', 30000.00, '2010-04-30', 'wushi@example.com', '13800138008');