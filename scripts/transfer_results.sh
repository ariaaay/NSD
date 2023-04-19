MIND_DIR="/user_data/yuanw3/project_outputs/NSD/output/encoding_results"
LOCAl_DIR="output/encoding_results"

MODELS="YFCC_clip \
YFCC_simclr \
YFCC_slip \
IC_title_clip \
IC_title_tag_description_clip
clip \
resnet50_bottleneck \
laion2b_clip \
laion400m_clip \
clip_visual_resnet
"

for subj in {1..8}; do
    echo $subj
    OUTDIR=$LOCAl_DIR/subj$subj
    for model in $MODELS; do
        INFILE=$MIND_DIR/subj$subj/rsq_${model}_whole_brain.p
        scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR 
    done

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_slip_YFCC_simclr_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR 

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_simclr_YFCC_clip_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_slip_clip_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR  

    INFILE=$MIND_DIR/subj$subj/rsq_laion2b_clip_laion400m_clip_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR  

    INFILE=$MIND_DIR/subj$subj/rsq_clip_laion400m_clip_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR  

    INFILE=$MIND_DIR/subj$subj/rsq_YFCC_clip_layer_n-1_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR

    INFILE=$MIND_DIR/subj$subj/rsq_clip_visual_resnet_resnet50_bottleneck_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR

    INFILE=$MIND_DIR/subj$subj/rsq_IC_title_clip_resnet50_bottleneck_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR

    INFILE=$MIND_DIR/subj$subj/rsq_IC_title_tag_description_clip_resnet50_bottleneck_whole_brain.p
    scp yuanw3@mind.cs.cmu.edu:$INFILE $OUTDIR
    

done