
# different combinations of conversations that can take place for table 1
# this prompt will help to find most suitable type of conversation
llm_selector_prompt_1 = """

        You are data analyst now given set of examples below choose the most appropriate prompt number
        according to {query}

        1 - [What is the economic classification of argentina,
            how is USA economically classified, Is india a developed country?, Is ghana a soliatry market?]
        2 - [How are Denmark,canda,autralian markets co-related,
            what is common point between Ireland and Australian markets,
            How are India and Chana related in terms of financial markets]
        3 - [when was India declared Emerging market, 
            on which date russia was given tag of solitary market]
        4 - [what are the list of countries declared as emerging markets,
            what is the list of countries classified as solitary markets and what is their count?
            what are list of countries that are not emerging,soliatry markets]
        5 - [which country was most recently declared emerging market,
            which country was most recently declared solitary market]
        6 - [which market first became developed market USA or australia,
            which market first became developed market USA or Chile]

        """