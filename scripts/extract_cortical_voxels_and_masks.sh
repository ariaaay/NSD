SUBJS="1 \
       2 \
       5 \
       7"

for subj in $SUBJS; do
  # python code/extract_cortical_voxel.py --subj $subj --mask_only

  # python code/extract_cortical_voxel.py --zscore_by_run --subj $subj

  # # extract ROI mask to apply on cortical data
  # python code/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-eccrois
  # python code/extract_cortical_voxel.py --subj $subj --mask_only --roi prf-visualrois
  # python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-faces
  # python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-words
  # python code/extract_cortical_voxel.py --subj $subj --mask_only --roi floc-places

  python code/compute_ev.py --subj $subj --zscored_input --compute_ev
done