from scripts.audit_rde_native_reference import native_source_equivalent


def test_only_checksum_helper_may_differ():
    old='import math\ndef _tree_digest(x):\n return str(x)\ndef native(x):\n return x+1\n'
    changed=old.replace('return str(x)','return repr(x)')
    assert native_source_equivalent(old,changed)
    assert not native_source_equivalent(old,old.replace('return x+1','return x+2'))
    assert not native_source_equivalent(old,old.replace('import math','import random'))
