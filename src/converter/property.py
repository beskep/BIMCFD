"""
test
"""
from typing import Optional

import ifcopenshell


def get_relating_property_def(entity: ifcopenshell.entity_instance):
  return entity.RelatingPropertyDefinition


def get_pset(entity: ifcopenshell.entity_instance,
             pset_name: Optional[str] = None):
  psets = [get_relating_property_def(x) for x in entity.IsDefinedBy]
  psets = [x for x in psets if x.is_a('IfcPropertySet')]
  if pset_name and psets:
    psets = [x for x in psets if x.Name.lower() == pset_name.lower()]

  return psets if psets else None


def get_pset_dict(pset: ifcopenshell.entity_instance):
  if not pset.is_a('IfcPropertySet'):
    raise ValueError

  prop = pset.HasProperties

  return {x.Name: str(x) for x in prop}
