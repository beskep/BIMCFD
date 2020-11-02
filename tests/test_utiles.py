import pytest

import utils


def test_utiles_path():
  assert utils.PRJ_DIR.exists()
  assert utils.SRC_DIR.exists()
  assert utils.SRC_DIR.name == 'src'
  assert utils.RESOURCE_DIR.exists()


if __name__ == "__main__":
  pytest.main(['-v', '-k', 'test_utiles'])
