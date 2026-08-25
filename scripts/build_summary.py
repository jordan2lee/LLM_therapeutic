#!/usr/bin/env python
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resp_file", required = False, default='results/responses_LLM.txt', help='input file with LLM output', type=str)
parser.add_argument("-p", "--prompt_file", required = False, default='results/prompts.txt', help='input file with prompts input to LLM', type=str)
parser.add_argument("-c", "--external_file", required = False, default='results/clinical_true.tsv', help='input file with external data to validate', type=str)
parser.add_argument("-o", "--outfile", required = True, help='output file with merged info', type=str)
args = parser.parse_args()

def parse_mutation_status(p_string):
    '''returns gene and mutated status (mutated, not mutated)'''
    match = re.search(r"'([A-Z0-9\-_]+)'\s+genomic mutation status", p_string, re.IGNORECASE)
    gene = match.group(1) if match else None
    
    match = re.search(r"genomic mutation status is\s+(.*?)\s+and", p_string, re.IGNORECASE)
    mutation_status = match.group(1).strip() if match else None
    
    match = re.search(r"protein expression is\s+(.*?)(?=\.|\s+Evaluate|$)", p_string, re.IGNORECASE)
    expression = match.group(1).strip() if match else None
    
    return gene, mutation_status, expression


# Merge multiple files into one single summary file
data = {'prompt':[], 'predicted_clinical_importance':[]}

with open(args.resp_file, 'r') as resp:
    for line in resp:
        r = line.strip()
        data['predicted_clinical_importance'].append(r)

# handle for subset analysis
max_i = len(data['predicted_clinical_importance'])
i=0
with open(args.prompt_file, 'r') as prompt:
    for line in prompt:
        if i<max_i:
            r = line.strip()
            data['prompt'].append(r)
            i +=1
        else:
            break
summary = pd.DataFrame(data)

# add info (gene, mutation) for the predicted vals
results = [parse_mutation_status(prompt) for prompt in summary['prompt']]
gene = []
mutation=[]
expr = []
for i in range(0,len(results)):
    g, m, e= results[i]
    gene.append(g)
    mutation.append(m)
    expr.append(e)
summary['gene']=gene
summary['mutation_status']=mutation
summary['gene_expression']=expr

# pull in outside curated lit
df=pd.read_csv(args.external_file, sep='\t', names = ['gene', 'gene_id', 'clinical_importance'])
gene2importance = dict(zip( df['gene'], df['clinical_importance']  ))
summary['clinical_importance'] = df['gene'].map(gene2importance)

summary.to_csv(args.outfile , sep='\t', index=False)
