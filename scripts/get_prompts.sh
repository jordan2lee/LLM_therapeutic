#!/bin/bash
GENES=($(awk -F'\t' 'NR==1 {for(i=1;i<=NF;i++) if($i=="gene") c=i; next} c {print $c}' results/reference_tcga.tsv | sort -u | awk '{printf "'\''%s'\'' ", $0}'))
for GENE in "${GENES[@]}"; do
    for MUTATION in 'mutated' 'not mutated'; do
        for EXPRESSION in 'over-expressed' 'under-expressed' 'normal'; do
            echo "Act as an expert clinical geneticist and variant curation officer. Evaluate the following molecular profile in a breast invasive carcinoma patient: ${GENE} genomic mutation status is ${MUTATION} and ${GENE} protein expression is ${EXPRESSION}. Evaluate these features together and choose one classification from ['Pathogenic', 'Likely pathogenic', 'Uncertain significance (VUS)', 'Likely benign', 'Benign']. Output ONLY the exact classification string and nothing else."
        done
    done
done