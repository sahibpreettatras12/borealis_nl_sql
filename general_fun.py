## Imports
# from api import get_completion,fn_agent_1
from langchain import PromptTemplate
import os
import psycopg2
import openai
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv,find_dotenv
from prompts_t1 import get_completion

# loadingd creds
_ = load_dotenv(find_dotenv())
open_ai_key = str(os.getenv('open_ai_key'))
postgres_host = str(os.getenv('postgres_host'))
postgres_database = str(os.getenv('postgres_database'))
user = str(os.getenv('user'))
password = str(os.getenv('password'))

#fetching open ai key
os.environ['OPENAI_API_KEY'] = open_ai_key
openai.api_key = open_ai_key

#defining postgres connection
con = psycopg2.connect(
            host=postgres_host,
            database=postgres_database,
            user=user,
            password=password)

#defining cursor
cursor = con.cursor()




# function used for Table 1
def fn_agent_1(user_query: str) -> str:
    
    # datafiles related to class 1
    sql_query_classification_update = 'SELECT * FROM "BGA_Dashboard_Reference_Database_Prod"."Ref_Country_Economic_Classification_Update_Table"'
    cursor.execute(sql_query_classification_update)
    records = cursor.fetchall()
    df1 = pd.DataFrame(records, columns=['Date','Country','Classification_Id'])
    
    
    #cursor to fetch dataframe Ref_Country_Economic_Classification_Update_Table.csv
    sql_query_classification = 'SELECT * FROM "BGA_Dashboard_Reference_Database_Prod"."Ref_Economic_Classification_Table"'
    cursor.execute(sql_query_classification)
    records = cursor.fetchall()
    
    df2 = pd.DataFrame(records, columns=['Classification_Id','Classification_Description','Is_Active'])

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0,openai_api_key=open_ai_key),  df = [df1,df2], verbose=True)
    try:
        return agent.run(user_query)
    except:
        return """Can you please rephrase the query I think i didn't get it"""

def final_answer_creator(structured_data,user_query):
    """
    Given a JSON containing steps to solve a problem now this function 
    will eventually solve the problem step by step and give a final result
    """

    #solve step1
    try:
        output_step1 = fn_agent_1(structured_data['step1'])
    except:
        output_step1=None
    
    #solve step2
    try:
        if structured_data['step2']:
            prompt2 = PromptTemplate.from_template(template=""" Given the information {output_step1} 
                                                   carefully follow {step2}""")
            prompt2 = prompt2.format(output_step1=output_step1,step2 = structured_data['step2'])
            output_step2 = fn_agent_1(prompt2)
        
        # if step2 was NONE then give a None output
        else:
            output_step2=None
    except:
        output_step2 = None

    #solve step3
    try:
        if structured_data['step3']:
            prompt3 = PromptTemplate.from_template(template=""" Give the information {output_step2} follow {step3}""")
            prompt3 = prompt3.format(output_step2=output_step2,step3 = structured_data['step3'])
            output_step3 = fn_agent_1(prompt3)
        
        # if step3 was NONE then give a None output
        else:
            output_step3 = None
    except:
        output_step3 = None

    try:
        if structured_data['step4']:
            prompt4 = PromptTemplate.from_template(template=""" Given the information {output_step3} carefully follow {step4}""")
            prompt4 = prompt4.format(output_step3=output_step3,step4 = structured_data['step4'])
            output_step4 = fn_agent_1(prompt4)
        
        # if step4 was NONE then give a None output
        else:
            output_step4=None
    except:
        output_step4 = None

    
    final_prompt = PromptTemplate.from_template(template=""" Give the information 
            {output_step1},{output_step2},{output_step3} follow {question} and 
            give a natural language response""")
    
    final_prompt = final_prompt.format(output_step1=output_step1,output_step2=output_step2,output_step3=output_step3,
                            question=user_query)
    
    # get the final result from completion API from openai
    result = get_completion(prompt=final_prompt,temperature=0.0)

    if type(result)==str:
        return result   
    
    else:
        return "Sorry can you please rephrase the output"