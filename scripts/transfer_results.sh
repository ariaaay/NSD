MIND_DIR="/user_data/yuanw3/project_outputs/NSD/output/encoding_results"
LOCAl_DIR="~/Projects/NSD/output/encoding_results/"

MODELS=(YFCC_clip \
YFCC_simclr \
YFCC_slip)

for subj in {1..8}; do
    OUTDIR=$LOCAl_DIR/subj$subj/
    for model in $MODELS; do
        INFILE=$MIND_DIR/subj$subj/rsq_${model}_whole_brain.p
        scp yuanw3@mind.cs.cmu.edu:$INFILE $OURDIR 
    done

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_slip_YFCC_simclr_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OURDIR 

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_simclr_YFCC_clip_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OURDIR 

done