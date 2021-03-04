source venv/bin/activate

NETWORKS="convnet_alexnet \
place_alexnet \
taskrepr_class_1000 \
taskrepr_class_places \
taskrepr_edge2d \
taskrepr_edge3d
"
for network in $NETWORKS; do
 python python code/run_cka.py --cka_across_networks_and_brain $network
done