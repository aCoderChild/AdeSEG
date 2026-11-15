from scripts.local_import_closure_v1 import local_import_closure


def test_discovers_absolute_from_and_relative_project_imports(tmp_path):
    for package in ('scripts', 'modeling'):
        (tmp_path/package).mkdir()
        (tmp_path/package/'__init__.py').write_text('')
    (tmp_path/'scripts/entry.py').write_text('from scripts import helper\nimport modeling.alpha\n')
    (tmp_path/'scripts/helper.py').write_text('from . import other\n')
    (tmp_path/'scripts/other.py').write_text('value = 1\n')
    (tmp_path/'modeling/alpha.py').write_text('from .beta import value\n')
    (tmp_path/'modeling/beta.py').write_text('value = 2\n')
    closure = local_import_closure(tmp_path, ['scripts/entry.py'])
    assert closure == {'scripts/entry.py', 'scripts/__init__.py', 'scripts/helper.py', 'scripts/other.py',
                       'modeling/__init__.py', 'modeling/alpha.py', 'modeling/beta.py'}
