from pathlib import Path

from s1_orbits import fetch_for_scene


def download_orbits(
    reference_scenes: list, secondary_scenes: list, orbit_directory: str = 'orbits'
) -> dict:

    orbit_dir = Path(orbit_directory)
    orbit_dir.mkdir(exist_ok=True)

    reference_orbits = {str(fetch_for_scene(scene, orbit_dir)) for scene in reference_scenes}
    secondary_orbits = {str(fetch_for_scene(scene, orbit_dir)) for scene in secondary_scenes}

    return {
        'orbit_directory': orbit_directory,
        'reference_orbits': list(reference_orbits),
        'secondary_orbits': list(secondary_orbits),
    }
