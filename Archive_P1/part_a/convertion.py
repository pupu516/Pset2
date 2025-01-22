import numpy as np

def coordinate_conversion():
    current_coor = input("What coordinate system are you in? cartesian/spherical/cylindrical \n").strip().lower()
    system = ['cartesian', 'spherical', 'cylindrical']
    
    if current_coor not in system:
        print('Error: Please select the option I listed above')
        return

    try:
        coor1 = float(input('Enter the first coordinate: '))
        coor2 = float(input('Enter the second coordinate: '))
        coor3 = float(input('Enter the third coordinate: '))
    except ValueError:
        print("Error: Please enter a valid number")
        return

    if current_coor == 'cartesian':
        r = np.linalg.norm([coor1, coor2, coor3])
        theta = np.arccos(coor3 / r)  
        phi = np.arctan2(coor2, coor1)
        sph = [r, theta, phi]
        print("Spherical coordinates: ", sph)
        
        r_ = np.linalg.norm([coor1, coor2])  
        theta_ = np.arctan2(coor2, coor1)  
        cyl = [r_, theta_, coor3]
        print("Cylindrical coordinates: ", cyl)

    elif current_coor == 'spherical':
        x = coor1 * np.sin(coor2) * np.cos(coor3)
        y = coor1 * np.sin(coor2) * np.sin(coor3)
        z = coor1 * np.cos(coor2)
        car = [x, y, z]
        print("Cartesian coordinates: ", car)
        
        r_ = coor1 * np.sin(coor2)  
        theta_ = coor3  
        cyl = [r_, theta_, z]
        print("Cylindrical coordinates: ", cyl)

    elif current_coor == 'cylindrical':

        x = coor1 * np.cos(coor2)
        y = coor1 * np.sin(coor2)
        z = coor3
        car = [x, y, z]
        print("Cartesian coordinates: ", car)
        
        r = np.sqrt(coor1**2 + coor3**2)  
        theta = np.arctan2(coor1, coor3) 
        phi = coor2  
        sph = [r, theta, phi]
        print("Spherical coordinates: ", sph)



def basis_conversion():
    return








