#assign_to_ref.py



from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import AgglomerativeClustering

import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import re


from sklearn.metrics.pairwise import cosine_similarity



data_folder = os.getcwd()

dd = pd.read_csv(data_folder + "reflist_labels_dedup.csv")

dd = dd[[ type(x)==str for x in dd.phil]]



model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    return mean_pooled.squeeze()  # -> 2D



print("calc embeddings for phil")
embeddings = torch.stack([get_embedding(text) for text in dd.phil])
print("done.")


#--- when there are multiple names for the same entity,
#    take the average of their emb
embeddings_array = embeddings.numpy()
labels = np.array(dd.label)

# Organize embeddings by cluster
cluster_embeddings = {}
for label, embedding in zip(labels, embeddings_array):
    if label not in cluster_embeddings:
        cluster_embeddings[label] = []
    cluster_embeddings[label].append(embedding)


average_embeddings = {}
for label, embeddings in cluster_embeddings.items():
    average_embeddings[label] = np.mean(embeddings, axis=0)



#----------------------------
#   v1
#---------------------------- 



modelnames = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro','llama2-70b']


curr_temp = "1" 


datalist_folder = data_folder + 'list_dfs/'



# first - organize from the lists to separate rows.

errs = []
rows = []


for promptversion in ['v1','v2','v3']:
    for whichmodel in modelnames[:]:
        print(whichmodel)
        #df = pd.read_parquet(datalist_folder + promptversion + "_"  +whichmodel + "_temp" + str(curr_temp)  +".parquet", engine='pyarrow')
        if whichmodel =='claude-3-sonnet':
            df = pd.read_parquet(datalist_folder + "claude-3-sonnet_temp1.0_" + promptversion +  ".parquet")
        else:
            try:
                df = pd.read_parquet(datalist_folder   +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion + ".parquet", engine='pyarrow')
            except:
                errs.append([whichmodel, promptversion])
                print("no parquet, try csv")
                continue
            #df = pd.read_parquet(datalist_folder   +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion + ".parquet", engine='pyarrow')
        print(len(df))
        print(df.columns)

        for i, row in df.iterrows():
            #print(row['region'])
            clist = row['list']
            for phil in clist:
                print(phil)
                phil = re.sub(r'\n', '', phil)
                phil = re.sub(r'\.\(\)[\s]', '', phil)
                phil = re.sub(r'^[\d]{1,2}\.?[\s]{0,2}', '', phil)
                rows.append([phil, whichmodel]) #, row['seed']])

    dd2 = pd.DataFrame(rows, columns=['phil','model']) # ,'region','seed'

    print(dd2.model.value_counts())


    # Now assign to nearest entity:
    errs = []
    rows = []


    for whichmodel in modelnames[:]:
        modelname = whichmodel
        dg = dd2[dd2.model==whichmodel]
        print("len: ", str(len(dg)))
        print("calc emb for new...")
        cand_embs = torch.stack([get_embedding(text) for text in dg.phil])
        print('done.')


        tensor_list = [torch.tensor(average_embeddings[label], dtype=torch.float32) for label in average_embeddings]
        average_embeddings_tensor = torch.stack(tensor_list)

        cand_embs_norm = cand_embs / cand_embs.norm(dim=1, keepdim=True)
        average_embeddings_norm = average_embeddings_tensor / average_embeddings_tensor.norm(dim=1, keepdim=True)

        similarity_matrix = cosine_similarity(cand_embs_norm, average_embeddings_norm)

        if not isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = torch.tensor(similarity_matrix)

        _, closest_indices = torch.max(similarity_matrix, dim=1)

        closest_labels = [labels[idx] for idx in closest_indices]
        print(closest_labels[50:70])
        dg['closest_label'] = closest_labels

        closest_file = data_folder + "closest/closest_" + modelname + "_" + promptname + ".csv"
        dg.to_csv(closest_file, index=False)



#----------------------------
#  v4, v5
#---------------------------- 



modelnames = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']

curr_temp = "1"


for promptversion in ['v4','v5']:
    rows = []

    for whichmodel in modelnames[:]:
        df = pd.read_parquet(data_folder + promptversion + "_"  +whichmodel + "_temp" + str(curr_temp)  +".parquet", engine='pyarrow')
        print(len(df))
        print(df.columns)

        for i, row in df.iterrows():
            #print(row['region'])
            clist = row['list']
            for phil in clist:
                print(phil)
                phil = re.sub(r'\n', '', phil)
                phil = re.sub(r'\.\(\)[\s]', '', phil)
                phil = re.sub(r'^[\d]{1,2}\.?[\s]{0,2}', '', phil)
                rows.append([phil, whichmodel]) #, row['seed']])

    dd2 = pd.DataFrame(rows, columns=['phil','model']) # ,'region','seed'


    for whichmodel in modelnames[:]:
        dg = dd2[dd2.model==whichmodel]


        print("calc emb for new...")
        cand_embs = torch.stack([get_embedding(text) for text in dg.phil])
        print('done.')

        tensor_list = [torch.tensor(average_embeddings[label], dtype=torch.float32) for label in average_embeddings]
        average_embeddings_tensor = torch.stack(tensor_list)

        # Normalize the embeddings to unit vectors to prepare for cosine similarity calculation
        cand_embs_norm = cand_embs / cand_embs.norm(dim=1, keepdim=True)
        average_embeddings_norm = average_embeddings_tensor / average_embeddings_tensor.norm(dim=1, keepdim=True)


        similarity_matrix = cosine_similarity(cand_embs_norm, average_embeddings_norm)

        if not isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = torch.tensor(similarity_matrix)

        _, closest_indices = torch.max(similarity_matrix, dim=1)


        closest_labels = [labels[idx] for idx in closest_indices]

        dg['closest_label'] = closest_labels

        closest_file = data_folder + "closest/closest_" + modelname + "_" + promptversion + ".csv"
        dg.to_csv(closest_file, index=False)


