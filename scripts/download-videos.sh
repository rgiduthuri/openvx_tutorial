# run this script from openvx_tutorial folder; the current C code requires
# tutorial_videos folder to be under the openvx_tutorial folder to be able
# pick the default video sequence when launching from Qt Creator
#
\rm -rf tutorial_videos
mkdir tutorial_videos
cd tutorial_videos
wget http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi
cd ..
