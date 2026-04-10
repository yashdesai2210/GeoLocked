import s2sphere

def CoordsToS2(lat, lon, level=12):
    # Converts GPS coordinates to a S2 Cell IDs
    # Takes 2d coords and plots onto sphere, turns sphere into flat faces on cube, zooms out guess to specified level, returns cell ID
    p1 = s2sphere.LatLng.from_degrees(lat, lon)
    cell = s2sphere.CellId.from_lat_lng(p1)
    cell_parent = cell.parent(level)
    return cell_parent.id()