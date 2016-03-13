# run this script from openvx_tutorial folder; the current C code requires
# tutorial_videos folder to be under the openvx_tutorial folder to be able
# pick the default video sequence when launching from Qt Creator
#
\rm -rf tutorial_videos.tgz tutorial_videos
wget http://ewh.ieee.org/r6/scv/sps/openvx-material/tutorial_videos.tgz
tar xvfz tutorial_videos.tgz
\rm -rf tutorial_videos.tgz
