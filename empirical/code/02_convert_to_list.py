#02_convert_to_list.py



from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

import pandas as pd
import time
import json
import re


list_phils_prompt = PromptTemplate(
    template="Identify ALL philosophical traditions or philosophers " \
        "specifically named or clearly implied in the input text, reporting just the name " \
        "of the school of thought or the philosopher. " \
        "If a philosopher or school of thought is clearly being referred to, " \
        "they may be included in the list.\n# Input text: {input_text}\n{format_instructions}",
    input_variables=["input_text"],
    partial_variables={"format_instructions": format_instructions},
)





model = ChatOpenAI(temperature=0)

chain = list_phils_prompt | model | output_parser


curr_temp = "1" #for input

data_folder =  os.getcwd()

errs = []

for promptversion in ['v1','v2_diverse','v3']:
    print('-----')
    for whichmodel in ["claude-3-sonnet",'gemini-pro','llama2-70b','gpt-3.5-turbo']:
        filein = data_folder +  whichmodel + "_temp" + curr_temp +"_" + promptversion+ '.csv'
        df = pd.read_csv(filein)
        print("len %d", len(df))
        texts = df.text
        print(promptversion, whichmodel)
        print(texts[1][-100:])


        lists = []

        meta_info = []

        i = start
        for intext in texts[0:100]:
            print(i)
            print(intext)
            i +=1
            try:
                res = chain.invoke({"input_text": intext})
                print(res)
                lists.append(res)
                meta_info.append([i, intext])
                time.sleep(1.1)
            except:
                print("failed")
                errs.append(i)
                time.sleep(3)

        print(errs)
        df2 = pd.DataFrame(meta_info)

        df2['lists'] = lists
        df2.columns = ['i', 'text', 'list']


        fileout = data_folder + whichmodel + "_temp" + str(curr_temp)  + '_' +promptversion +".csv"
        df2.to_csv(fileout, index=False)
        df2.to_parquet(data_folder + whichmodel + "_temp" + str(curr_temp) +'_'+ promptversion +".parquet", engine='pyarrow')







