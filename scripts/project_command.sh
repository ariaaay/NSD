subj=1

python code/process_permuation_results.py --subj $subj
python code/run_significance_test.py --subj $subj --use_empirical_p
python code/make_task_matrix.py --subj $subj --method "cosine" --use_mask_corr --empirical
python code/make_task_tree.py --method masked_corr --subj $subj