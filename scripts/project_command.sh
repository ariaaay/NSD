subj=1

python code/extract_image_list.py --subj $subj --type trial
python code/extract_cortical_voxel.py --zscored_by_run --subj $subj

python code/compute_ev.py --subj $subj --zscored_input
. scripts/run_encoding_models_on_taskonomy_tayer.sh $subj

python code/process_permuation_results.py --subj $subj

python code/run_significance_test.py --subj $subj --use_empirical_p

python code/make_task_matrix.py --subj $subj --method "cosine" --use_mask_corr --empirical
# if need to exclude certain ROI:
python code/make_task_matrix.py --subj $subj --method "cosine" --use_mask_corr --empirical --exclude_roi prf-visualrois

python code/make_task_tree.py --method masked_corr --subj $subj
python code/make_task_tree.py --method masked_corr --subj $subj --exclude_roi prf-visualrois