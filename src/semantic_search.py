from sentence_transformers import SentenceTransformer, util
import torch
import json
import jsonlines
from tqdm import tqdm 

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# query_file_path = '/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/alpaca_instruction_data/self_instruct_jsonlines.json'

corpus_file_path = '/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/dolly-human/dolly-15k.jsonl'

query_text_key = 'instruction'
corpus_text_key = 'instruction'

def read_data(file_path, text_key='instruction'):
    datas = []
    lines = []
    with jsonlines.open(file_path, 'r') as f:
        # data = json.load(f)
        for line in f:
            datas.append(line[text_key])
            lines.append(line)
    return datas, lines

corpus, corpus_lines = read_data(corpus_file_path, corpus_text_key)
# queries = read_data(query_file_path, query_text_key)
queries = [
    'You are given a review of movie. Your task is to classify given movie review into two categories: 1) positive, and 2) negative based on its content.',
    'In this task, you are provided with an article of the legal acts. Your task is to classify it into three categories (Regulation, Decision and Directive) based on its content: 1) Regulation is a binding legislative act that must be applied in its entirety on a set date across all the member states (European Union countries). 2) Decision is binding on those to whom it is addressed (e.g. an European Union country or an individual company) and is directly applicable. 3) Directive is a legislative act that sets out a goal that all must achieve. However, it is up to the individual countries to devise their own laws on how to reach these goals.',
    'Given a sentence in English, provide an equivalent paraphrased version from the original that retains the same meaning.',
    'Given a sentence in English, provide an equivalent paraphrased translation in Chinese that retains the same meaning both through the translation and the paraphrase',
    'Please translate the following sentence from Chinese into English.',
    ]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
output = []
top_k = min(10, len(corpus))
for query in tqdm(queries, desc='quering...'):
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in Instructions:")
    cur_sims = []
    for score, idx in zip(top_results[0], top_results[1]):
        # print(corpus[idx], "(Score: {:.4f})".format(score))
        cur_sims.append({f'{idx}':corpus[idx], 'score':"{:.4f}".format(score.item())})
        
    output.append({f'query_{query_text_key}':query, f'top-{top_k}_sims':cur_sims})

output_path = '/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/dolly-human/query_sim.json'
# import pdb; pdb.set_trace()
json.dump(output, open(output_path, 'w+'), indent=4)

"""
# Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
    print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
"""