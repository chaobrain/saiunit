# build SAIUnit
python setup.py bdist_wheel
cd dist/
twine upload saiunit*

# build BrainUnit
cd ./brainunit
python make_brainunit_code.py
python setup.py bdist_wheel
cd dist/
twine upload brainunit*

