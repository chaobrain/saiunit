import os

curdir, _ = os.path.split(__file__)
units_fname = os.path.normpath(os.path.join(curdir, '../saiunit/_unit_common.py'))

_siprefixes = {
  "y": 1e-24,
  "z": 1e-21,
  "a": 1e-18,
  "f": 1e-15,
  "p": 1e-12,
  "n": 1e-9,
  "u": 1e-6,
  "m": 1e-3,
  "c": 1e-2,
  "d": 1e-1,
  "": 1,
  "da": 1e1,
  "h": 1e2,
  "k": 1e3,
  "M": 1e6,
  "G": 1e9,
  "T": 1e12,
  "P": 1e15,
  "E": 1e18,
  "Z": 1e21,
  "Y": 1e24
}

fundamental_units = ['metre', 'meter', 'kilogram', 'second', 'amp', 'ampere', 'kelvin', 'mole', 'mol', 'candle']

#### DERIVED UNITS, from http://physics.nist.gov/cuu/Units/units.html
# columns: long name, symbol, dimension, auto_simplify, unicode, latex
derived_unit_table = [
  ['radian', 'rad', 'get_or_create_dimension()', False, None, None],
  ['steradian', 'sr', 'get_or_create_dimension()', False, None, None],
  ['hertz', 'Hz', 'get_or_create_dimension(s=-1)', True, None, None],
  ['newton', 'N', 'get_or_create_dimension(m=1, kg=1, s=-2)', True, None, None],
  ['pascal', 'Pa', 'get_or_create_dimension(m=-1, kg=1, s=-2)', True, None, None],
  ['joule', 'J', 'get_or_create_dimension(m=2, kg=1, s=-2)', True, None, None],
  ['watt', 'W', 'get_or_create_dimension(m=2, kg=1, s=-3)', True, None, None],
  ['coulomb', 'C', 'get_or_create_dimension(s=1, A=1)', True, None, None],
  ['volt', 'V', 'get_or_create_dimension(m=2, kg=1, s=-3, A=-1)', True, None, None],
  ['farad', 'F', 'get_or_create_dimension(m=-2, kg=-1, s=4, A=2)', True, None, None],
  ['ohm', 'ohm', 'get_or_create_dimension(m=2, kg=1, s=-3, A=-2)', True, 'Ω', r'\Omega'],
  ['siemens', 'S', 'get_or_create_dimension(m=-2, kg=-1, s=3, A=2)', True, None, None],
  ['weber', 'Wb', 'get_or_create_dimension(m=2, kg=1, s=-2, A=-1)', False, None, None],
  ['tesla', 'T', 'get_or_create_dimension(kg=1, s=-2, A=-1)', False, None, None],
  ['henry', 'H', 'get_or_create_dimension(m=2, kg=1, s=-2, A=-2)', False, None, None],
  ['lumen', 'lm', 'get_or_create_dimension(cd=1)', False, None, None],
  ['lux', 'lx', 'get_or_create_dimension(m=-2, cd=1)', False, None, None],
  ['becquerel', 'Bq', 'get_or_create_dimension(s=-1)', False, None, None],
  ['gray', 'Gy', 'get_or_create_dimension(m=2, s=-2)', False, None, None],
  ['sievert', 'Sv', 'get_or_create_dimension(m=2, s=-2)', False, None, None],
  ['katal', 'kat', 'get_or_create_dimension(s=-1, mol=1)', False, None, None],
]

## Generate derived unit objects and make a table of base units from these and the fundamental ones
base_units = fundamental_units + ['kilogramme', 'gram', 'gramme', 'molar']
derived = ''
for longname, shortname, definition, auto, uni, latex in derived_unit_table:
  kwargs = ''
  if auto:
    kwargs += ', auto_simplify=True'
  if uni is not None:
    kwargs += ', unicode="{}", latex=r"{}"'.format(uni, latex)
  derived += '{longname} = Unit.create({definition}, "{longname}", "{shortname}"{kwargs})\n'.format(
    longname=longname, shortname=shortname, definition=definition, kwargs=kwargs)
  base_units.append(longname)


all_units = base_units + []

definitions = '######### SCALED BASE UNITS ###########\n'

# Generate scaled units for all base units
scaled_units = []
excluded_scaled_units = set()
# Prefixed kelvin units are exported but not powered (no "mkelvin2") and not
# part of the display lists below.
unpowered_exports = []
for _bu in base_units:
  if _bu in ['kelvin', 'kilogram', 'kilogramme']:
    continue  # do not create "kkilogram" etc.; kelvin is handled after candle
  for _k in _siprefixes.keys():
    if len(_k):
      _u = _k + _bu
      all_units.append(_u)
      definitions += '{_u} = Unit.create_scaled_unit({_bu}, "{_k}")\n'.format(
        _u=_u, _bu=_bu, _k=_k)
      if _k not in ["da", "d", "c", "h"]:
        scaled_units.append(_u)
      else:
        excluded_scaled_units.add(_u)
  if _bu == 'candle':
    for _k in _siprefixes.keys():
      if len(_k):
        _u = _k + 'kelvin'
        unpowered_exports.append((len(all_units), _u))
        definitions += '{_u} = Unit.create_scaled_unit(kelvin, "{_k}")\n'.format(_u=_u, _k=_k)

# Generate 2nd and 3rd powers for all scaled base units
definitions += '######### SCALED BASE UNITS TO POWERS ###########\n'
powered_units = []
for bu in all_units + []:
  for i in [2, 3]:
    u = bu + str(i)
    definitions += '{u} = Unit.create_power({bu}, {i}, name="{u}")\n'.format(u=u, bu=bu, i=i)
    all_units.append(u)
    if bu not in excluded_scaled_units:
      powered_units.append(u)

    additional_units = '''
# Current list from http://physics.nist.gov/cuu/Units/units.html, far from complete
additional_units = [
    pascal * second, newton * metre, watt / metre ** 2, joule / kelvin,
    joule / (kilogram * kelvin), joule / kilogram, watt / (metre * kelvin),
    joule / metre ** 3, volt / metre ** 3, coulomb / metre ** 3, coulomb / metre ** 2,
    farad / metre, henry / metre, joule / mole, joule / (mole * kelvin),
    coulomb / kilogram, gray / second, katal / metre ** 3,
    # We don't want liter/litre to be used as a standard unit for display, so we
    # put it here instead of in the standard units
    '''

# Scaled units for liter (we don't want liter**2, etc., so it is not included
# in the base units above; also, we don't want it to register in the standard
# units, so it is not chosen instead of meter**3 for displaying volume
# quantities)
for _bu in ['liter', 'litre']:
  all_units.append(_bu)
  for _k in _siprefixes.keys():
    _u = _k + _bu
    all_units.append(_u)
    definitions += '{_u} = Unit.create_scaled_unit({_bu}, "{_k}")\n'.format(_u=_u, _bu=_bu, _k=_k)
    additional_units += _u + ', '
additional_units += ']'


modular_dynamics_units = [
  'eV', 'Angstrom', 'AMU', 'IMF'
]

# all_units = all_units + modular_dynamics_units


# Add unit names to __all__ (deduplicated, kelvin prefixes inserted in place)
export_names = list(all_units)
for _pos, _u in reversed(unpowered_exports):
  export_names.insert(_pos, _u)
all = '''
__all__ = [
{allunits}
    ]
'''.format(allunits='\n'.join('    "' + u + '",' for u in dict.fromkeys(export_names)))


def to_definition(name, x):
  return '''
{name} = [
{items}
    ]
'''.format(name=name, items='\n'.join('    ' + i + ',' for i in x))


# Note that for presentation purposes, the `UnitRegistry` uses the *last*
# defined unit for a given scale. Therefore, we reserve the order of the units
# so that the standard units come first.
with open('units_template.py') as f:
  template = f.read()
units_str = template.format(
  derived=derived, all=all, definitions=definitions,
  base_units=to_definition('base_units', base_units[::-1]),
  scaled_units=to_definition('scaled_units', scaled_units[::-1]),
  powered_units=to_definition('powered_units', powered_units[::-1]),
  all_units=to_definition('all_units', all_units[::-1]),
  additional_units=additional_units,
)

with open(units_fname, 'w') as f:
  f.write(units_str)
