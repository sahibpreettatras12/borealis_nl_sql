from api import get_completion
from langchain import PromptTemplate

def steps_table1_from_prompts(prompt_number,user_query):
    if prompt_number==2:
        prompt = PromptTemplate.from_template(template="""COnsider yourself Data Anlayst now Answer the user query.
            and remeber to keep instructions in mind
            1. Figure out country or group of countries which have been talked about in {query}.
            2. Check for the economic classification id of country or group of countries.
            3. Answer the query based on user data.
        """)
        prompt = prompt.format(query=user_query)
        
        
        final_prompt =PromptTemplate.from_template(
            template="""Given a set of instructions in braces {temp_prompt} figure out different steps 
                        that can be taken your response should be json and in 3 steps
                        step1:___,
                        step2:___,
                        step3:___
                        
                        ---
                        """)
        
        final_prompt = final_prompt.format(temp_prompt=prompt)
        
        result = get_completion(prompt=final_prompt,temperature=0.0)

        return result
    
    elif prompt_number==4:
        prompt = PromptTemplate.from_template(template="""COnsider yourself Data Anlayst now Answer the user query.
            and remeber to keep instructions in mind
            1. Figure out how countries associated to econimic classification mentioned in {query}.
            2. Now after looking all data find a approriate operation to reach to answer for {query}  
            3. Answer the query based on user data.
            
        """)
        prompt = prompt.format(query=user_query)
        
        
        final_prompt =PromptTemplate.from_template(
            template="""Given a set of instructions in braces {temp_prompt} figure out different steps 
                        that can be taken your response should be json and in 3 steps
                        step1:___,
                        step2:___,
                        step3:___
                        
                        ---
                        """)
        
        final_prompt = final_prompt.format(temp_prompt=prompt)
        
        result = get_completion(prompt=final_prompt,temperature=0.0)

        return result
    
    elif prompt_number==5:
        prompt = PromptTemplate.from_template(template="""COnsider yourself Data Anlayst now Answer the user query.
            and remeber to keep instructions in mind
            1. Figure out country or group of countries which have been talked about in {query}
            2.Check only for the economic classification id of country or group of countries.
            3. Sort them according to date and answer the query.
        """)
        prompt = prompt.format(query=user_query)
        
        
        final_prompt =PromptTemplate.from_template(
            template="""Given a set of instructions in braces {temp_prompt} figure out different steps 
                        that can be taken your response should be json and in 3 steps
                        step1:___,
                        step2:___,
                        step3:___
                        
                        ---
                        """)
        
        final_prompt = final_prompt.format(temp_prompt=prompt)
        
        result = get_completion(prompt=final_prompt,temperature=0.0)

        return result
    
    elif prompt_number==6:
        prompt = PromptTemplate.from_template(template="""COnsider yourself Data Anlayst now Answer the user query.
            and remeber to keep instructions in mind
            1.  Figure out country or group of countries which have been talked about in {query}..
            2. Check for the economic classification id of country or group of countries.
            3. Answer the query based on user data.
        """)
        prompt = prompt.format(query=user_query)
        
        
        final_prompt =PromptTemplate.from_template(
            template="""Given a set of instructions in braces {temp_prompt} figure out different steps 
                        that can be taken your response should be json and in 3 steps
                        step1:___,
                        step2:___,
                        step3:___
                        
                        ---
                        """)
        
        final_prompt = final_prompt.format(temp_prompt=prompt)
        
        result = get_completion(prompt=final_prompt,temperature=0.0)

        return result
    
