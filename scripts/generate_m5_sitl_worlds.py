#!/usr/bin/env python3
"""Generate three reproducible, non-corridor M5 SITL worlds from PX4 default."""

from pathlib import Path
import os
import xml.etree.ElementTree as ET


def add_cylinder(world, name, x, y, radius, height):
    model = ET.SubElement(world, 'model', {'name': name})
    ET.SubElement(model, 'static').text = 'true'
    link = ET.SubElement(model, 'link', {'name': 'link'})
    collision = ET.SubElement(link, 'collision', {'name': 'collision'})
    geometry = ET.SubElement(collision, 'geometry')
    cylinder = ET.SubElement(geometry, 'cylinder')
    ET.SubElement(cylinder, 'radius').text = str(radius)
    ET.SubElement(cylinder, 'length').text = str(height)
    visual = ET.SubElement(link, 'visual', {'name': 'visual'})
    visual_geometry = ET.SubElement(visual, 'geometry')
    visual_cylinder = ET.SubElement(visual_geometry, 'cylinder')
    ET.SubElement(visual_cylinder, 'radius').text = str(radius)
    ET.SubElement(visual_cylinder, 'length').text = str(height)
    material = ET.SubElement(visual, 'material')
    ET.SubElement(material, 'ambient').text = '0.20 0.45 0.75 1'
    ET.SubElement(material, 'diffuse').text = '0.20 0.45 0.75 1'
    ET.SubElement(model, 'pose').text = f'{x} {y} {height / 2} 0 0 0'


def main():
    px4 = Path(os.environ.get('PX4_DIR', Path.home() / 'projects/PX4-Autopilot'))
    source = px4 / 'Tools/simulation/gz/worlds/default.sdf'
    if not source.is_file():
        raise SystemExit(f'PX4 default Gazebo world not found: {source}')
    output_dir = Path(__file__).resolve().parents[1] / 'worlds'
    output_dir.mkdir(parents=True, exist_ok=True)
    designs = {
        'm5_obstacle_world_1': [(2.4, 0.0, 0.42, 2.2)],
        'm5_obstacle_world_2': [(2.3, 1.0, 0.38, 2.4), (5.0, -0.9, 0.45, 2.0)],
        'm5_obstacle_world_3': [(2.0, -0.9, 0.40, 2.4), (4.1, 1.1, 0.42, 2.2), (6.0, -0.6, 0.35, 1.8)],
    }
    for world_name, obstacles in designs.items():
        tree = ET.parse(source)
        root = tree.getroot()
        world = root.find('world')
        if world is None:
            raise SystemExit(f'No <world> element in {source}')
        world.set('name', world_name)
        for idx, (x, y, radius, height) in enumerate(obstacles, 1):
            add_cylinder(world, f'{world_name}_obstacle_{idx}', x, y, radius, height)
        target = output_dir / f'{world_name}.sdf'
        tree.write(target, encoding='utf-8', xml_declaration=True)
        print(f'Generated {target}')


if __name__ == '__main__':
    main()
