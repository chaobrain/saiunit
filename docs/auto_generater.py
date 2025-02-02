# -*- coding: utf-8 -*-

import importlib
import inspect
import os

block_list = ['test', 'register_pytree_node', 'call', 'namedtuple', 'jit', 'wraps', 'index', 'function']


def get_class_funcs(module):
    classes, functions, others = [], [], []
    # Solution from: https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]
    for k in names:
        data = getattr(module, k)
        if not inspect.ismodule(data) and not k.startswith("_"):
            if inspect.isfunction(data):
                functions.append(k)
            elif isinstance(data, type):
                classes.append(k)
            else:
                others.append(k)

    return classes, functions, others


def _write_module(module_name, automodule, filename, header=None, template=False):
    module = importlib.import_module(module_name)
    classes, functions, others = get_class_funcs(module)

    fout = open(filename, 'w')
    # write header
    if header is None:
        header = f'``{module_name}`` module'
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {automodule} \n')
    fout.write(f'.. automodule:: {automodule} \n\n')

    # write autosummary
    fout.write('.. autosummary::\n')
    if template:
        fout.write('   :template: classtemplate.rst\n')
    fout.write('   :toctree: generated/\n\n')
    for m in functions:
        fout.write(f'   {m}\n')
    for m in classes:
        fout.write(f'   {m}\n')
    for m in others:
        fout.write(f'   {m}\n')

    fout.close()


def _write_submodules(module_name, filename, header=None, submodule_names=(), section_names=()):
    fout = open(filename, 'w')
    # write header
    if header is None:
        header = f'``{module_name}`` module'
    else:
        header = header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {module_name} \n')
    fout.write(f'.. automodule:: {module_name} \n\n')

    # whole module
    for i, name in enumerate(submodule_names):
        module = importlib.import_module(module_name + '.' + name)
        classes, functions, others = get_class_funcs(module)

        fout.write(section_names[i] + '\n')
        fout.write('-' * len(section_names[i]) + '\n\n')

        # write autosummary
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')

        fout.write(f'\n\n')

    fout.close()


def _write_subsections(module_name,
                       filename,
                       subsections: dict,
                       header: str = None):
    fout = open(filename, 'w')
    header = f'``{module_name}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {module_name} \n')
    fout.write(f'.. automodule:: {module_name} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, values in subsections.items():
        fout.write(name + '\n')
        fout.write('-' * len(name) + '\n\n')
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in values:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _write_subsections_v2(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{out_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {out_path} \n')
    fout.write(f'.. automodule:: {out_path} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, subheader in subsections.items():
        module = importlib.import_module(f'{module_path}.{name}')
        classes, functions, others = get_class_funcs(module)

        fout.write(subheader + '\n')
        fout.write('-' * len(subheader) + '\n\n')
        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _write_subsections_v3(module_path,
                          out_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{out_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')
    fout.write(f'.. currentmodule:: {out_path} \n')
    fout.write(f'.. automodule:: {out_path} \n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 2' + '\n\n')

    for section in subsections:
        fout.write(subsections[section]['header'] + '\n')
        fout.write('-' * len(subsections[section]['header']) + '\n\n')

        fout.write(f'.. currentmodule:: {out_path}.{section} \n')
        fout.write(f'.. automodule:: {out_path}.{section} \n\n')

        for name, subheader in subsections[section]['content'].items():
            module = importlib.import_module(f'{module_path}.{section}.{name}')
            classes, functions, others = get_class_funcs(module)

            fout.write(subheader + '\n')
            fout.write('~' * len(subheader) + '\n\n')
            fout.write('.. autosummary::\n')
            fout.write('   :toctree: generated/\n')
            fout.write('   :nosignatures:\n')
            fout.write('   :template: classtemplate.rst\n\n')
            for m in functions:
                fout.write(f'   {m}\n')
            for m in classes:
                fout.write(f'   {m}\n')
            for m in others:
                fout.write(f'   {m}\n')
            fout.write(f'\n\n')

    fout.close()


def _write_subsections_v4(module_path,
                          filename,
                          subsections: dict,
                          header: str = None):
    fout = open(filename, 'w')
    header = f'``{module_path}`` module' if header is None else header
    fout.write(header + '\n')
    fout.write('=' * len(header) + '\n\n')

    fout.write('.. contents::' + '\n')
    fout.write('   :local:' + '\n')
    fout.write('   :depth: 1' + '\n\n')

    for name, (subheader, out_path) in subsections.items():

        module = importlib.import_module(f'{module_path}.{name}')
        classes, functions, others = get_class_funcs(module)

        fout.write(subheader + '\n')
        fout.write('-' * len(subheader) + '\n\n')

        fout.write(f'.. currentmodule:: {out_path} \n')
        fout.write(f'.. automodule:: {out_path} \n\n')

        fout.write('.. autosummary::\n')
        fout.write('   :toctree: generated/\n')
        fout.write('   :nosignatures:\n')
        fout.write('   :template: classtemplate.rst\n\n')
        for m in functions:
            fout.write(f'   {m}\n')
        for m in classes:
            fout.write(f'   {m}\n')
        for m in others:
            fout.write(f'   {m}\n')
        fout.write(f'\n\n')

    fout.close()


def _get_functions(obj):
    return set([n for n in dir(obj)
                if (n not in block_list  # not in blacklist
                    and callable(getattr(obj, n))  # callable
                    and not isinstance(getattr(obj, n), type)  # not class
                    and n[0].islower()  # starts with lower char
                    and not n.startswith('__')  # not special methods
                    )
                ])


def _import(mod, klass=None, is_jax=False):
    obj = importlib.import_module(mod)
    if klass:
        obj = getattr(obj, klass)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
    else:
        if not is_jax:
            return obj, ':obj:`{}.{{}}`'.format(mod)
        else:
            from docs import implemented_jax_funcs
            return implemented_jax_funcs, ':obj:`{}.{{}}`'.format(mod)


def main(package: str):
    os.makedirs('apis/', exist_ok=True)

    assert package in ['brainunit', 'saiunit']

    # _write_module(module_name='saiunit.math._einops',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.einops.rst',
    #               header='Einstein Operations',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._fun_array_creation',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.array-creation.rst',
    #               header='Array Creation Functions',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._fun_accept_unitless',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.unitless.rst',
    #               header='Functions that Accepting Unitless',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._fun_change_unit',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.change-unit.rst',
    #               header='Functions that Changing Unit',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._fun_keep_unit',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.keep-unit.rst',
    #               header='Functions that Keeping Unit',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._fun_remove_unit',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.remove-unit.rst',
    #               header='Functions that Removing Unit',
    #               template=True)
    #
    # _write_module(module_name='saiunit.math._misc',
    #               automodule='saiunit.math',
    #               filename='apis/saiunit.math.misc.rst',
    #               header='Other Functions',
    #               template=True)

    module_and_name = [
        ('_activation', 'Activation Functions'),
        ('_alias', 'Unit Processing'),
        ('_einops', 'Einstein Operations'),
        ('_fun_accept_unitless', 'Functions that Accepting Unitless'),
        ('_fun_array_creation', 'Array Creation Functions'),
        ('_fun_change_unit', 'Functions that Changing Unit'),
        ('_fun_keep_unit', 'Functions that Keeping Unit'),
        ('_fun_remove_unit', 'Functions that Removing Unit'),
        ('_misc', 'Other Functions'),
    ]

    _write_submodules(module_name=f'{package}.math',
                      filename=f'apis/{package}.math.rst',
                      header=f'``{package}.math`` module',
                      submodule_names=[k[0] for k in module_and_name],
                      section_names=[k[1] for k in module_and_name])

    module_and_name = [
        ('_linalg_change_unit', 'Functions that Changing Unit'),
        ('_linalg_keep_unit', 'Functions that Keeping Unit'),
    ]

    _write_submodules(
        module_name=f'{package}.linalg',
        filename=f'apis/{package}.linalg.rst',
        header=f'``{package}.linalg`` module',
        submodule_names=[k[0] for k in module_and_name],
        section_names=[k[1] for k in module_and_name]
    )

    module_and_name = [
        ('_lax_accept_unitless', 'Functions that Accepting Unitless'),
        ('_lax_array_creation', 'Array Creation Functions'),
        ('_lax_change_unit', 'Functions that Changing Unit'),
        ('_lax_keep_unit', 'Functions that Keeping Unit'),
        ('_lax_remove_unit', 'Functions that Removing Unit'),
        ('_lax_linalg', 'Linalg Functions'),
        ('_misc', 'Other Functions'),
    ]

    _write_submodules(
        module_name=f'{package}.lax',
        filename=f'apis/{package}.lax.rst',
        header=f'``{package}.lax`` module',
        submodule_names=[k[0] for k in module_and_name],
        section_names=[k[1] for k in module_and_name]
    )

    module_and_name = [
        ('_fft_change_unit', 'Functions that Changing Unit'),
        ('_fft_keep_unit', 'Functions that Keeping Unit'),
    ]

    _write_submodules(
        module_name=f'{package}.fft',
        filename=f'apis/{package}.fft.rst',
        header=f'``{package}.fft`` module',
        submodule_names=[k[0] for k in module_and_name],
        section_names=[k[1] for k in module_and_name]
    )


if __name__ == '__main__':
    main('brainunit')
