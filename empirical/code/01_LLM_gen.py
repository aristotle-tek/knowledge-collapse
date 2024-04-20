



import json
import pandas as pd
from openai import OpenAI
import time



if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


import requests
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import asyncio


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OPENAI Key")


openai_client = OpenAI()


prompt_v1 = "What does the well-being of a human depend on according to philosophers?"
prompt_v2 = "What does the well-being of a human depend on according to diverse philosophers?"
prompt_v3 = "What does the well-being of a human depend on? Provide as many diverse philosophers and philosophic traditions as possible."


def openai_call(prompt, seed, model):
    response = openai_client.chat.completions.create(
          logprobs=model=='gpt-3.5-turbo',
          model=model,
          seed=seed,
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
          ]
        )
    return response



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__


whichmodel = "gpt-3.5-turbo"
curr_temp = "1" # chatgpt default

data_folder = os.getcwd()



responses = []
texts = []
errs = []


for currprompt in [prompt_v1, prompt_v2, prompt_v3]:
    responses = []
    texts = []
    for i in range(0, 100):
        print(i)
        try:
            response = openai_call(currprompt, i, whichmodel)
            response_json = json.dumps(response, cls=CustomEncoder)
            txt = response.choices[0].message.content
            print(txt)
            responses.append(response_json)
            texts.append(txt)
        except:
            print('failed.')
            errs.append(i)
            time.sleep(2)
        time.sleep(1.1)

    print("errs: ", str(errs))

    df = pd.DataFrame(texts)
    df['req_json'] = responses
    df.columns = ['text', 'req_json']
    fileout = data_folder + whichmodel + "_temp" + str(curr_temp)  +".csv"
    df.to_csv(fileout, index=False)



#----------------------------
# prompt v4
#----------------------------

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()




list_phils_prompt = PromptTemplate(
    template="Suggest philosophical perspectives, traditions or schools " \
        "of thought on the question:  {question}\n" \
        "reporting a list of {number} names of philosophers who " \
        "address the question. " \
        "\n{format_instructions}",
    input_variables=["question", "number"],
    partial_variables={"format_instructions": format_instructions},
)




promptversion = 'v4_list20'
whichmodel = "gpt-3.5-turbo"
curr_temp = "1" # chatgpt default


responses = []
lists = []
errs = []


start = 0
end = 100


for i in range(start, end):
    print(i)
    try:
        if whichmodel[:5]=='gpt-3':
            model = ChatOpenAI(model=whichmodel, temperature=curr_temp, model_kwargs={"seed":currseed}) # default gpt-3.5-turbo
        elif whichmodel=='gemini-pro':
            model = ChatGoogleGenerativeAI(model="gemini-pro")#, temperature=curr_temp)#,  model_kwargs={"seed":i})
        elif whichmodel[:13]=='claude-3-sonn':
            model = ChatAnthropic(model="claude-3-sonnet-20240229")

        chain = list_phils_prompt | model | output_parser
        response = chain.invoke({"question": question, "number": "twenty"})
        response_json = json.dumps(response, cls=CustomEncoder)
        # response is directly the list....# response.choices[0].message.content
        print(response)
        responses.append(response_json)
        lists.append(response)
    except:
        print('failed.')
        errs.append(i)
        time.sleep(2)
    time.sleep(1.1)

print(errs)


df = pd.DataFrame(responses)

df['list'] = lists

df.columns = [ 'req_json','list']

fileout = data_folder + promptversion + "_"  +whichmodel + "_temp" + str(curr_temp)  +".csv"
df.to_csv(fileout, index=False)

df.to_parquet(data_folder + promptversion + "_"  +whichmodel + "_temp" + str(curr_temp)  +".parquet", engine='pyarrow')



#----------------------------
# prompt v5
#----------------------------


with open(data_folder + 'attributes_regions_34.txt', 'r') as file:
    regions_list = [line.strip() for line in file]



promptversion = 'v5_list20_regions'
whichmodel = "gpt-3.5-turbo"
curr_temp = "1" # chatgpt default


list_phils_prompt_attributes = PromptTemplate(
    template="Create a CSV list of philosophical perspectives, traditions or schools " \
        "of thought from {attribute} on the question:  {question}\n" \
        "reporting a CSV list of up to {number} names of philosophers who directly " \
        "address the question. " \
        "\n{format_instructions}",
    input_variables=["attribute", "question", "number"],
    partial_variables={"format_instructions": format_instructions},
)





metadata = []
#texts = []
lists = []
errs = []


question = 'What does the well-being of a human depend on?'



for currseed in [0,1,2]:
    for i in range(0, len(regions_list)):
        print(i)
        try:
            model = ChatOpenAI(model=whichmodel, temperature=curr_temp, model_kwargs={"seed":currseed}) # default gpt-3.5-turbo
            chain = list_phils_prompt_attributes | model | output_parser
            curr_region = regions_list[i]
            response = chain.invoke({'attribute': curr_region, "question": question, "number": "twenty"})
            lists.append(response)
            metadata.append([curr_region, currseed])
        except:
            print('failed.')
            errs.append(i)
            time.sleep(2)
        time.sleep(1.1)


print(errs)


dfrg = pd.DataFrame(metadata, columns=['region','seed'])


# correction - for when does not correctly format lists with comma sep
corrected_lists = []
for lst in lists:
    if len(lst)==1:
        print(lst)
        sp = lst[0].split('\n')
        print(len(sp))
        sp2 = [re.sub(r'^[\d]{1,2}\.?[\s]{0,2}', '', x) for x in sp]
        corrected_lists.append(sp2)
    else:
        corrected_lists.append(lst)

for lst in corrected_lists:
    if len(lst)!=20:
        print(len(lst))
        print(lst)



dfrg['list'] = corrected_lists

fileout = data_folder + promptversion + "_"  +whichmodel + "_temp" + str(curr_temp)  +".csv"
dfrg.to_csv(fileout, index=False)



















