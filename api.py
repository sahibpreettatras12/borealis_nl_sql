# Using flask to make an api
# import necessary libraries and functions
import torch
import os
import psycopg2
import openai
import pandas as pd
from pydantic import BaseModel, Field
import pydantic_chatcompletion
from llm_selector_prompt_tables import llm_selector_prompt_1
from flask import Flask, jsonify, request
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv,find_dotenv
from prompts_t1 import steps_table1_from_prompts
from general_fun import final_answer_creator,fn_agent_1

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


#sbert for converting sentences to embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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
embeddings = sentence_model.encode(sentences)


def get_completion(prompt, model="gpt-3.5-turbo",temperature=0):
    messages = [{"role": "user", "content": "You are a data anlayst who has to answer questions from data using user query"}
               ,{"role":"user","content":prompt}]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# Define the pydantic model
class Prompt_Selector(BaseModel):
    prompt_name: int = Field(description="Ideal prompt number selected for the problem")
    



# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/single/',methods =['POST'])
def agent_classifier_function():
    
    payload = request.get_json()
    print(sentence_model)
    try:
        user_query = payload['user_query']
    except:
        return jsonify({'message':'Please ask a question'})
    query_embedding = sentence_model.encode(user_query)
    tensor_of_all_similarities = util.dot_score(query_embedding, embeddings)
    index_of_most_similar = int(torch.argmax(tensor_of_all_similarities))
    label_classified = df.iloc[index_of_most_similar]['label']

    if label_classified == 1:

        
        query = llm_selector_prompt_1.format(query=user_query)
       # Set up messages
        messages = [
            {"role": "user", "content": "Logically look and think about problem and find most similar prompt number"},
            {"role": "user", "content": query},
        ]

        # Use pydantic_chatcompletion to get a structured data class
        selected_prompt = pydantic_chatcompletion.create(messages, Prompt_Selector, model='gpt-3.5-turbo')
        selected_prompt = selected_prompt.prompt_name
        print("Selected_prompts  ---->",selected_prompt)


        # if prompts 1 and 3 are selected
        if selected_prompt in [1,3]:
            nl_response = fn_agent_1(user_query+'re check your answer based on data provided')
            return jsonify({'Natural_response':natural_response})
        else:

            model_name = "gpt-3.5-turbo-0613"
            temperature = 0.0
            model = OpenAI(model_name=model_name, temperature=temperature)

            # result we will get would be steps to solve the problem
            structured_data = steps_table1_from_prompts (selected_prompt,user_query)
            print("structured_data  --->",structured_data)
            if structured_data!=0:
                # Number of steps to solve the problem
                number_of_steps = len(structured_data)

                # to get the final answer in a natural language like response
                natural_response = final_answer_creator(structured_data,user_query)
            
                return jsonify({'Natural_response':natural_response})

# driver function
if __name__ == '__main__':
  
    app.run(debug = True)