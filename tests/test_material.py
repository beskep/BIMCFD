import os

import pytest
from converter import material_match as mat


def kor_eng_material():
  """테스트용 한국어-영어 재료 목록 생성

  Returns
  -------

  """
  import json

  kor_eng = {
      '알루미늄': 'aluminum',
      '콘크리트': 'concrete',
      '콘크리트 블록': 'concrete block',
      '철': 'iron',
      '스테인리스': 'stainless',
      '유리섬유': 'glass fiber',
      '유리': 'glass'
  }
  # kor_eng_json = json.dumps(kor_eng, indent=4, ensure_ascii=False)
  # with open(mat._KOR_ENG_PATH, 'w', encoding='utf-8-sig') as f:
  #   f.write(kor_eng_json)
  return kor_eng


@pytest.fixture
def material():
  return mat.MaterialMatch()


def test_has_korean(material: mat.MaterialMatch):
  assert material.has_korean('한글')
  assert not material.has_korean('alphabet')
  assert material.has_korean('!@#^#나asf2ㅎ9ㅕ5') == 3


def test_kor_eng(material: mat.MaterialMatch):
  assert isinstance(material._kor_eng, dict)
  assert material._kor_eng['유리'] == 'glass'
  assert material._kor_eng['콘크리트'] == 'concrete'


def test_friction_with_english_material(material: mat.MaterialMatch):
  material_name = 'concrete'
  roughness, name, nearest_name, score = material.friction(material_name)

  assert isinstance(roughness, float)
  assert roughness

  assert isinstance(name, str)
  assert name == material_name

  assert isinstance(nearest_name, str)
  assert nearest_name


def test_friction_with_korean_material(material: mat.MaterialMatch):
  material_name = '콘크리트'
  roughness, name, nearest_name, score = material.friction(material_name)

  assert isinstance(roughness, float)
  assert roughness

  assert isinstance(name, str)
  assert name == 'concrete'

  assert isinstance(nearest_name, str)
  assert nearest_name


def test_thermal_properties_from_db(material: mat.MaterialMatch):
  material_name = '1-dodecene'
  prop, eng_name, nearest_name = material.thermal_properties(material_name)

  assert material_name == eng_name
  assert nearest_name == material_name

  assert prop['rho'] == 761
  assert prop['Cp'] == 2150
  assert prop['k'] == 0.14


def test_thermal_properties_from_ht(material: mat.MaterialMatch):
  material_name = 'Metals, stainless steel '
  prop, eng_name, nearest_name = material.thermal_properties(material_name)

  assert eng_name == 'Metals, stainless steel '
  assert nearest_name == 'Metals, stainless steel'

  assert prop['rho'] == 7900
  assert prop['Cp'] == 460
  assert prop['k'] == 17


def test_material_list():
  from ht import insulation
  from fluids import friction
  from pathlib import Path

  roughness = list(friction._all_roughness.keys())
  thermal = list(insulation.materials_dict.keys())

  result_dir = Path(r'D:\CFD\test\material')
  if not result_dir.exists():
    os.makedirs(str(result_dir))

  with open(result_dir / 'dict_roughness.txt', 'w') as f:
    f.write('\n'.join(roughness))
  with open(result_dir / 'dict_thermal.txt', 'w') as f:
    f.write('\n'.join(thermal))


if __name__ == '__main__':
  # kor_eng_material()
  pytest.main(['-v', '-k', 'test_material.py'])
