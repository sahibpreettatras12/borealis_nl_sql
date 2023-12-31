# Using flask to make an api
# import necessary libraries and functions
import torch
import os
import psycopg2
import openai
import pandas as pd

from flask import Flask, jsonify, request
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


from dotenv import load_dotenv,find_dotenv

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

# creating a Flask app
app = Flask(__name__)
  
# dataframe containg supervised question and their labels 
df =  pd.read_csv('borealis-questions.csv')
sentences = df['questions'].tolist()

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)


# function used for Table 1
def fn_agent_1(user_query: str) -> str:
    
    # datafiles related to class 1
    # df1= pd.read_csv('table_1/Ref_Country_Economic_Classification_Update_Table.csv')
    # df2 = pd.read_csv('table_1/Ref_Economic_Classification_Table.csv')


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



# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/single/',methods =['POST'])
def agent_classifier_function():
    
    payload = request.get_json()
    try:
        user_query = payload['user_query']
    except:
        return jsonify({'message':'Please ask a question'})
    query_embedding = model.encode(user_query)
    tensor_of_all_similarities = util.dot_score(query_embedding, embeddings)
    index_of_most_similar = int(torch.argmax(tensor_of_all_similarities))
    label_classified = df.iloc[index_of_most_similar]['label']
    if label_classified == 1:
        nl_returned = fn_agent_1(user_query)
        
        return jsonify({'Natural_response':nl_returned})

# driver function
if __name__ == '__main__':
  
    app.run(debug = True)