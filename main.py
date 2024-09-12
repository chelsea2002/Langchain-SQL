# 配置数据库相关内容
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
import re



import json
# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate

# example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
# examples_prompt_fewer_samples = [
#     {
#         "input": "一共有多少国家？",
#         "query": "SELECT COUNT(DISTINCT country_name) AS total_countries FROM Countries;"
#     },
#     {
#         "input":""
#     }
# ]


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# prompt_fewer_samples = FewShotPromptTemplate(
#     examples = examples_prompt_fewer_samples,
#     example_prompt=example_prompt,
#     prefix="""You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. 
#     Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}
#     \n\nBelow are a number of examples of questions and their corresponding SQL queries.""",
#     suffix="User input: {input}\nSQL query: ",
#     input_variables=["input", "top_k", "table_info"],
# )

database = SQLDatabase.from_uri("sqlite:///C:\\Users\Administrator\Desktop\临时文件\\test_demo\\backend\db\coal.db")
chain = create_sql_query_chain(llm, database)  #prompt_fewer_samples

# prompt_validation = ChatPromptTemplate.from_messages(
#         [("system", system_validation), ("human", "{query}")]
#     ).partial(dialect=database.dialect)

# validation_chain = prompt_validation | llm | StrOutputParser()

# full_chain = {"query": chain} | validation_chain

with open('test_coal.json','r',encoding='utf-8') as file:
    data = json.load(file)

train_sql = []
golden_sql = []

for item in data:
    question = item['question']
    sqlquery = chain.invoke(
            {
                "question": question
            }
        )
    extracted_sqlquery = re.findall(r'SELECT[\s\S]*?(?=```)' , sqlquery)[0]
    train_sql.append(extracted_sqlquery)
    


with open('train.sql', 'w', encoding='utf-8') as file:
    for sql in train_sql:
        file.write(sql)


# 打开和读取黄金SQL文件
with open('golden_sql.sql', 'r', encoding='utf-8') as file:
     # 读取所有行并去除空白字符
    lines = file.readlines()

    # 遍历每一行，去除首尾空白字符并添加到列表
    for line in lines:
    # 去除空白和换行符，并过滤掉空行
        cleaned_line = line.strip()
        if cleaned_line:  # 如果行不为空
            golden_sql.append(cleaned_line)



# exact match accuracy
def compare_sql(train_sql_list, golden_sql_list):
    total_score = 0
    # 遍历train_sql和golden_sql的每一项，比较是否完全相同
    for train, golden in zip(train_sql_list, golden_sql_list):
        if train.strip() == golden.strip():  # 使用strip()去除前后空格
            total_score += 1  # 匹配则加1分
    
    # 计算EM：总得分除以SQL个数，得到匹配比例
    em_score = total_score / len(golden_sql_list)
    return em_score

# 调用比较函数
em_score = compare_sql(train_sql, golden_sql)




print(f"EM得分 (Exact Match Score): {em_score:.2f}")







