
import csv
import numpy as np

human_emb_path = "data/human_similarity_judgement/spose_embedding_49d_sorted.txt"
human_emb = np.loadtxt(human_emb_path)

output_file = "data/human_similarity_judgement/spose_embedding_49d_sorted.tsv"
with open(output_file, 'w', encoding='utf8', newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    for emb in human_emb:
        tsv_writer.writerow(emb)