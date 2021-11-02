subj=1
# extract trial ID list
python code/extract_image_list.py --subj $subj --type trial
python code/extract_image_list.py --subj $subj --type cocoId

# prepare brain voxels for encoding models:
#   - extract cortical mask;
#   - mask volume metric data;
#   - zscore data by runs
python code/extract_cortical_voxel.py --zscore_by_run --subj $subj

# extract ROI mask to apply on cortical data
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-eccrois
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-visualrois
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-faces
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-words
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-places
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-bodies
python code/extract_cortical_voxel.py --subj $subj --mask_only --roi Kastner2015


# computer explainable variance for the data and output data averaged by repeats
python code/compute_ev.py --subj $subj --zscored_input


#=====================lines below is not done on NSD==========================
# run encoding models on taskonomy (on tayer)
. scripts/run_encoding_models_on_taskonomy_tayer.sh $subj

python code/process_permuation_results.py --subj $subj

python code/run_significance_test.py --subj $subj --use_empirical_p

python code/make_task_matrix.py --subj $subj --method "cosine" --use_mask_corr --empirical
# if need to exclude certain ROI:
python code/make_task_matrix.py --subj $subj --method "cosine" --use_mask_corr --empirical --exclude_roi prf-visualrois

python code/make_task_tree.py --method masked_corr --subj $subj
python code/make_task_tree.py --method masked_corr --subj $subj --exclude_roi prf-visualrois