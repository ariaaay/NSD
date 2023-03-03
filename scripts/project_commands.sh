for subj in {1..8}; do
    echo "processing subj $subj"
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
    python code/extract_cortical_voxel.py --subj $subj --mask_only --roi HCP_MMP1


    # computer explainable variance for the data and output data averaged by repeats
    python code/compute_ev.py --subj $subj --zscored_input
done

python code/analyze_clip_results.py --performance_analysis_by_roi --group_analysis_by_roi --summary_statistics --clip_rsq_across_subject

for subj in 1 2 3 4 6 7 8; do
    echo "processing subj $subj"
    python code/analyze_clip_results.py --process_bootstrap_results --subj $subj
done
