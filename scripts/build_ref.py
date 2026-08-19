#!/usr/bin/env python
import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputfile", required = False, default='src/TMP_20230209/BRCA_v12_20210228.tsv', help='input file to use as reference for LLM benchmarking', type=str)
parser.add_argument("-o", "--outputfile", required = False, default='results/reference_tcga.tsv', help='output tsv file name', type=str)
args = parser.parse_args()


def expression_info(val):
    if pd.isna(val):
        return 'missing'
    elif val>=2:
        return 'overexpressed'
    elif val<=-2:
        return 'underexpressed'
    else:
        return 'normal'

df = pd.read_csv(args.inputfile, sep='\t', index_col=0,low_memory=False)

# Filter for mutation fts and clean up naming
keep_muts= [c for c in df.columns if c.startswith('B:MUTA')]
muts_df = df[keep_muts]
mut_short = [c.strip().split(':')[3] for c in keep_muts]
muts_df.columns=mut_short

# Filter for gene expr fts and clean up naming
keep_expr= [c for c in df.columns if c.startswith('N:GEXP') and "?" not in c]
expr_df = df[keep_expr]
expr_short = [c.strip().split(':')[3] for c in keep_expr]
expr_df.columns=expr_short
# # remove genes where no expression over all samples 
expr_df =expr_df.loc[:, expr_df.sum(axis=0) > 20]

# Retain only fts with gexp and mut measured
shared_fts = list(set(muts_df.columns).intersection(set(expr_df.columns)))
shared_samples = list(set(muts_df.index).intersection(set(expr_df.index)))

muts_df = muts_df.loc[shared_samples,shared_fts]
muts_df = muts_df.loc[:, ~muts_df.columns.duplicated(keep='first')]

expr_df=expr_df.loc[shared_samples,shared_fts]
expr_df = expr_df.loc[:, ~expr_df.columns.duplicated(keep='first')]

# Gene expr in z-score
col_labels = expr_df.columns
index_labels = expr_df.index
z_expr_df= pd.DataFrame(stats.zscore(expr_df),columns = col_labels, index=index_labels)

# Build summary table
results = []
for sample in shared_samples:
    for ft in shared_fts:
        muta=muts_df.loc[sample, ft]
        gexp=z_expr_df.loc[sample, ft]
        results.append(
            {
                'sample':sample,
                'gene':ft,
                'mutation': muta,
                'expression':gexp
            }
        )
summary = pd.DataFrame(results)
summary["expression_status"] = summary["expression"].apply(expression_info)

summary.to_csv(args.outputfile, sep='\t', index=False)
