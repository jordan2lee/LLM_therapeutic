#!/usr/bin/env python

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputfile", required = True, help='input summary file of expression and mutations', type=str)
args = parser.parse_args()

summary = pd.read_csv(args.inputfile, sep='\t')

# Compare for overexpressed genes
s1 = summary[summary['gene_expression']=='over-expressed']
s1 = s1[s1['mutation_status']=='mutated'].reset_index(drop=True)
assert s1.shape[0]==len(set(s1['gene'])), 'need to dedup genes'
# harmonize
s1['predicted_clinical_importance'] = s1['predicted_clinical_importance'].str.replace('Uncertain significance (VUS)', 'Uncertain significance', regex=False)
y_true = list(s1['clinical_importance'])
y_pred = list(s1['predicted_clinical_importance'])
ba = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average = 'macro', zero_division=0.0)
recall = recall_score(y_true, y_pred, average= 'macro', zero_division=0.0)
print('Over-expressed genes with mutations:')
print(f'{ba:.2f} balanced accuracy, {acc:.2f} accuracy, {precision:.2f} precision, {recall:.2f} recall')

# Compare for under genes
s1 = summary[summary['gene_expression']=='under-expressed']
s1 = s1[s1['mutation_status']=='mutated'].reset_index(drop=True)
assert s1.shape[0]==len(set(s1['gene'])), 'need to dedup genes'
# harmonize
s1['predicted_clinical_importance'] = s1['predicted_clinical_importance'].str.replace('Uncertain significance (VUS)', 'Uncertain significance', regex=False)
y_true = list(s1['clinical_importance'])
y_pred = list(s1['predicted_clinical_importance'])
ba = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average = 'macro', zero_division=0.0)
recall = recall_score(y_true, y_pred, average= 'macro', zero_division=0.0)
print('Under-expressed genes with mutations:')
print(f'{ba:.2f} balanced accuracy, {acc:.2f} accuracy, {precision:.2f} precision, {recall:.2f} recall')
