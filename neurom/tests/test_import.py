from nose import tools as ntools

def test_import_neurom():
    try:
        import neurom
        return True
    except Exception:
        return False
