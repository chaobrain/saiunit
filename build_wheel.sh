# build SAIUnit
python setup.py bdist_wheel
twine upload dist/saiunit*

# build BrainUnit
python make_brainunit_setup.py
cd ./brainunit
python setup.py bdist_wheel
twine upload dist/brainunit*

