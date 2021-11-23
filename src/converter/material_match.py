import json
import re

from utils import DIR

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extractOne

from fluids import friction
from ht import insulation

DB_PATH = DIR.RESOURCE.joinpath('material.csv')
KOR_ENG_PATH = DIR.RESOURCE.joinpath('material_kor_eng.json')


def load_kor_eng(path):
  with open(path, 'r', encoding='utf-8-sig') as f:
    kor_eng = json.load(f)
  return kor_eng


def match_insulation(material, prop):
  if prop not in ['rho', 'Cp', 'k']:
    raise ValueError

  if prop == 'rho':
    fn = insulation.rho_material
  elif prop == 'Cp':
    fn = insulation.Cp_material
  else:
    fn = insulation.k_material

  try:
    res = fn(material)
  except ValueError:
    res = None

  return res


class MaterialMatch:

  def __init__(self, db_path=None, kor_eng_path=None):
    if db_path is None:
      db_path = DB_PATH
    if kor_eng_path is None:
      kor_eng_path = KOR_ENG_PATH

    self._db = pd.read_csv(db_path, engine='python', na_values='-')
    self._names = list(self._db['Name'])
    self._korean_pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]')
    self._kor_eng: dict = load_kor_eng(kor_eng_path)
    self._scorer = fuzz.token_set_ratio

  @property
  def database(self):
    return self._db

  @property
  def kor_eng_dict(self):
    return self._kor_eng

  @property
  def scorer(self):
    return self._scorer

  def has_korean(self, string):
    count = len(self._korean_pattern.findall(string))
    return count

  def to_english_material(self, material_name: str):
    if self.has_korean(material_name):
      matched_korean, score = extractOne(query=material_name,
                                         choices=self._kor_eng.keys(),
                                         scorer=self._scorer)
      eng_name = self._kor_eng[matched_korean]
    else:
      eng_name = material_name
      score = None

    return eng_name, score

  def friction(self, material_name: str):
    """ 이름이 가장 유사한 재료의 조도 반환
    입력값이 False인 경우 모든 입력값 None
    
    Parameters
    ----------
    material_name: str
      재료 이름

    Returns
    -------
    tuple: tuple containing:
      roughness: float, 조도 [m]
      english_name: str, 영어 재료명 (입력값이 영어인 경우 동일)
      nearest_name: str, DB와 매칭된 재료명
      score: int, 영어 재료명의 매칭 점수 (0-100)

    """
    if not material_name:
      return None, None, None, None

    eng_name, eng_score = self.to_english_material(material_name)

    nearest_material = friction.nearest_material_roughness(eng_name)
    roughness = friction.material_roughness(nearest_material)
    score = self._scorer(eng_name, nearest_material)

    return roughness, eng_name, nearest_material, score

  def thermal_properties(self, material_name: str):
    """이름이 가장 유사한 재료의 열적 특성 반환

    Parameters
    ----------
    material_name: str
      재료 이름

    Returns
    -------
    tuple: tuple containing:
      properties: dict,
        {'rho': density [kg/m^3],
         'Cp': specific heat [J/kgK],
         'k': conductivity [W/mK]}
      english_name: str, 영어 재료명 (입력값이 영어인 경우 동일)
      nearest_name: str, DB와 매칭된 재료명
    """
    eng_name, score = self.to_english_material(material_name)

    nearest_ht = insulation.nearest_material(eng_name)
    score_ht = self._scorer(eng_name, nearest_ht)

    nearest_db, score_db = extractOne(query=eng_name,
                                      choices=self._names,
                                      scorer=self._scorer)

    if score_ht >= score_db:
      prop = {x: match_insulation(nearest_ht, x) for x in ['rho', 'Cp', 'k']}
      nearest_name = nearest_ht
    else:
      series = self._db.loc[self._db['Name'] == nearest_db, :].squeeze()
      prop = {
          'rho': series['Density {kg/m3}'],
          'Cp': series['Specific Heat {J/kg-K}'],
          'k': series['Conductivity {W/m-K}']
      }
      nearest_name = nearest_db

    return prop, eng_name, nearest_name
