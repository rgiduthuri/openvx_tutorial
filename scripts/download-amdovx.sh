# download and extract amdovx-core/openvx into tutorial_exercises folder
#
\rm -rf master.zip tutorial_exercises/amdovx-core
wget https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-core/archive/master.zip
echo creating tutorial_exercises/amdovx-core ...
unzip -q master.zip
mv amdovx-core-master tutorial_exercises/amdovx-core
\rm -rf master.zip
