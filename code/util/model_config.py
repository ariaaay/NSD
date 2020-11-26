taskrepr_features = [
    "autoencoder",
    "denoise",
    "colorization",
    "curvature",
    "edge2d",
    "edge3d",
    "keypoint2d",
    "keypoint3d",
    "segment2d",
    "segment25d",
    "segmentsemantic",
    "rgb2depth",
    "rgb2mist",
    "reshade",
    "rgb2sfnorm",
    "room_layout",
    "vanishing_point",
    "class_1000",
    "class_places",
    # "jigsaw",
    "inpainting_whole",
]

convnet_structures = ["res50", "vgg16"]

model_features = dict()
model_features["taskrepr"] = taskrepr_features
model_features["convnet"] = convnet_structures


task_label = {
    "class_1000": "Object Class",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    "rgb2sfnorm": "Normals",
    "rgb2depth": "Depth",
    "rgb2mist": "Distance",
    "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    "autoencoder": "Autoencoding",
    "colorization": "Color",
    "edge3d": "Occlusion Edges",
    "edge2d": "2D Edges",
    "denoise": "Denoising",
    "curvature": "Curvature",
    "class_places": "Scene Class",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    "segment2d": "2D Segm.",
    # "jigsaw": "Jigsaw",
    "inpainting_whole": "Inpainting",
}

task_label_NSD_tmp = {
    "class_1000": "Object Class",
    "segment25d": "2.5D Segm.",
    "room_layout": "Layout",
    # "rgb2sfnorm": "Normals",
    # "rgb2depth": "Depth",
    # "rgb2mist": "Distance",
    # "reshade": "Reshading",
    "keypoint3d": "3D Keypoint",
    "keypoint2d": "2D Keypoint",
    # "autoencoder": "Autoencoding",
    # "colorization": "Color",
    "edge3d": "Occlusion Edges",
    "edge2d": "2D Edges",
    # "denoise": "Denoising",
    # "curvature": "Curvature",
    "class_places": "Scene Class",
    "vanishing_point": "Vanishing Pts.",
    "segmentsemantic": "Semantic Segm.",
    # "segment2d": "2D Segm.",
    # "jigsaw": "Jigsaw",
    "inpainting_whole": "Inpainting",
}


task_label_in_Taskonomy19_matrix_order = {
    "class_1000": "Object Class",
    "segment2d": "2D Segm.",
    "segmentsemantic": "Semantic Segm.",
    "class_places": "Scene Class",
    "denoise": "Denoising",
    "edge2d": "2D Edges",
    "edge3d": "Occlusion Edges",
    # "jigsaw": "Jigsaw",
    "autoencoder": "Autoencoding",
    "colorization": "Color",
    "keypoint3d": "3D Keypoint",
    "reshade": "Reshading",
    "rgb2mist": "Distance",
    "rgb2depth": "Depth",
    "rgb2sfnorm": "Normals",
    "room_layout": "Layout",
    "segment25d": "2.5D Segm.",
    "keypoint2d": "2D Keypoint",
    "inpainting_whole": "Inpainting",
    "curvature": "Curvature",
    "vanishing_point": "Vanishing Pts.",
}

conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
fc_layers = ["fc6", "fc7"]

ecc_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "0.5 deg",
    2: "1 deg",
    3: "2 deg",
    4: "4 deg",
    5: ">4 deg",
}

visual_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "V1v",
    2: "V1d",
    3: "V2v",
    4: "V2d",
    5: "V3v",
    6: "v3d",
    7: "h4v",
}

place_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OPA",
    2: "PPA",
    3: "RSC",
}

face_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OFA",
    2: "FFA-1",
    3: "FFA-2",
    4: "mTL-faces",
    5: "aTL-faces",
}

word_roi_names = {
    -1: "non_cortical",
    0: "cortical",
    1: "OWFA",
    2: "VWFA-1",
    3: "VWFA-2",
    4: "mfs-words",
}
