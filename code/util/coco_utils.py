from pycocotools.coco import COCO

trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
train_caps = COCO(trainFile)
val_caps = COCO(valFile)

def load_captions(cid):
    annIds = train_caps.getAnnIds(imgIds=[cid])
    anns = train_caps.loadAnns(annIds)
    if anns == []:
        annIds = val_caps.getAnnIds(imgIds=[cid])
        anns = val_caps.loadAnns(annIds)

    if anns == []:
        print("no captions extracted for image: " + str(cid))

    captions = [d["caption"] for d in anns]
    return captions