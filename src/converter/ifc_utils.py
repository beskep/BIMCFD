from typing import Optional

from ifcopenshell import entity_instance as IfcEntity


def entity_name(entity: IfcEntity) -> str:
  if hasattr(entity, 'LongName') and entity.LongName:
    name = f'{entity.Name} ({entity.LongName})'
  else:
    name = str(entity.Name)

  return name


def get_storey(entity):
  contained_in_structure = entity.ContainedInStructure
  if len(contained_in_structure) == 0:
    return None

  assert len(contained_in_structure) == 1
  storey = contained_in_structure[0].RelatingStructure
  assert storey.is_a('IfcBuildingStorey')

  return storey


def get_bounded_by(space: IfcEntity):
  if not space.is_a('IfcSpace'):
    raise ValueError(f'Need IfcSpace, not {space.is_a()}')

  try:
    boundaries: Optional[list] = [
        x.RelatedBuildingElement for x in space.BoundedBy
    ]
  except AttributeError:
    boundaries = None

  return boundaries
