# build SAIUnit
python -m build
# twine upload dist/saiunit*

# build BrainUnit
python make_brainunit_setup.py
cd ./brainunit
python -m build
# twine upload dist/brainunit*

