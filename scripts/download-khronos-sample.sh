# download and extract openvx_sample into tutorial_exercises folder
#
\rm -rf openvx_sample_1.0.1.tar.bz2 tutorial_exercises/openvx_sample
wget https://www.khronos.org/registry/vx/sample/openvx_sample_1.0.1.tar.bz2
echo creating tutorial_exercises/openvx_sample ...
bunzip2 -c openvx_sample_1.0.1.tar.bz2 | tar xf -
mv openvx_sample tutorial_exercises/openvx_sample
\rm -rf openvx_sample_1.0.1.tar.bz2
