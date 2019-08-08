import csv
from math import radians, cos, sin, asin, sqrt, atan2, degrees


def coordDiffFromCSV(config, A1, A2,
                     coordFile='../DRAOJunJul2018/Data/coordinates.csv'):
    """
    Returns distance in meters and angle in degrees (from A1 clockwise to
    A2) between gps points.

    Parameters
    ----------
    config : int
        Which array configuration to consider. Options are '1' or 2.
    A1 : int
        Antenna number (physical label). 0 for Tx.
    A2 : int
        Antenna number (physical label). 0 for Tx.
    coordFile : string
        CSV file with the coordinates in it.
    """

    A1Lat = 0.0
    A2Lat = 0.0
    A1Lon = 0.0
    A2Lon = 0.0

    config = 'Config' + str(config)
    if A1 == 0:
        A1 = 'Tx'
    elif A1 != 0:
        A1 = '#' + str(A1)
    if A2 == 0:
        A2 = 'Tx'
    elif A2 != 0:
        A2 = '#' + str(A2)

    # Read file
    csvfile = open(coordFile, 'rt')
    contents = csv.reader(csvfile, delimiter=',')
    for row in contents:
        if row[0] == config:
            if row[3] == A1:
                A1Lat = float(row[1])
                A1Lon = float(row[2])
            elif row[3] == A2:
                A2Lat = float(row[1])
                A2Lon = float(row[2])

    csvfile.close()

    # Check that the requested antennas were in the file
    try:
        5/A1Lat  # 5 is meaningless
    except ZeroDivisionError:
        print('Ant1 is not in {}'.format(config))
        return

    try:
        5/A2Lat  # 5 is meaningless
    except ZeroDivisionError:
        print('Ant2 is not in {}'.format(config))
        return

    # Put everything in radians & useful deltas
    A1Lon, A1Lat, A2Lon, A2Lat = map(radians, [A1Lon, A1Lat, A2Lon, A2Lat])
    dLon = A2Lon - A1Lon
    dLat = A2Lat - A1Lat

    # Calculate distance
    a = sin(dLat/2)**2 + cos(A1Lat) * cos(A2Lat) * sin(dLon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in meters
    dist = c * r  # Distance between antennas in meters (float)

    # Calculate bearing
    x = sin(dLon) * cos(A2Lat)
    y = cos(A1Lat) * sin(A2Lat) - (sin(A1Lat) * cos(A2Lat) * cos(dLon))

    initial_bearing = atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180deg to + 180deg which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return (dist, compass_bearing)
