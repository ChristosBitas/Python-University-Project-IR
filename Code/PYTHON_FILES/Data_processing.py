import pandas as pd
import os

docData=[]
docName=[]
# read directory of txt data
for file in os.listdir("_data/"):
    if not file.endswith("txt"):
        continue
    with open("_data/"+file,encoding="UTF-8") as infile:
        text = " ".join(infile.readlines()[:1000])
        docData.append(text)
        docName.append(file)
# store the text data and docId in a dataframe
df = pd.DataFrame()
df["doc"] = docName
df["text"] = docData

df.to_csv("_data/text_data.csv", encoding="utf-8", index=False) # store the dataframe in a csv for future use