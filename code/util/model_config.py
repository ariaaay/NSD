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
