# build SAIUnit
python setup.py bdist_wheel
cd ./dist/
#twine upload saiunit*
cd ../

# build BrainUnit
python make_brainunit_code.py
cd ./brainunit
python setup.py bdist_wheel
cd ./dist/
#twine upload brainunit*

