# from numpy import array
# import pyny3d.geoms as pyny

# coords_3d = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.7071067811865475, 0.7071067811865476], [2.0, 0.7071067811865475, 0.7071067811865476], [2.0, 1.414213562373095, 1.4142135623730951], [0.0, 1.414213562373095, 1.4142135623730951]])

# polygon = pyny.Polygon(coords_3d)
# print(f'Area is : {polygon.get_area()}')

def det(a):
    '''
    Determinant of matrix a, used to find the normal vector of the plane defined by three points
    '''
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

def unit_normal(a, b, c):
    '''
    Unit normal vector of plane defined by points a, b, and c; uses the determinant function to find the components of the normal vector and then normalizes it
    '''
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

def dot(a, b):
    '''
    The dot product of vectors a and b, it is used to project the total cross product vector onto the unit normal vector
    '''
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    '''
    The cross product of vectors a and b, used to find the area vector of the parallelogram spanned by two vectors
    '''
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

def area(poly):
    '''
    Calculates the area of a polygon in 3D space by summing up the cross products of consecutive edges, 
    projecting this sum onto the unit normal vector of the plane, and taking half of the absolute value of this projection
    '''
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

if __name__ == '__main__':
    poly = [[0, 0, 0], [7, 0, 0], [7, 10, 2], [0, 10, 2]]
    print("Area:", area(poly))
