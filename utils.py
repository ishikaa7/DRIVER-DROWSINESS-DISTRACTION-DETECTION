import math

def get_distance(p1, p2):
    #Calculates the Euclidean distance between two points
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_ear(eye_points, mesh_points):
    
    v1 = get_distance(mesh_points[eye_points[12]], mesh_points[eye_points[4]])
    v2 = get_distance(mesh_points[eye_points[11]], mesh_points[eye_points[5]])
    h = get_distance(mesh_points[eye_points[0]], mesh_points[eye_points[8]])
    return (v1 + v2) / (2.0 * h) # ear -> Eye Aspect Ratio

def get_mar(mesh_points):
    # mar -> Mouth Aspect Ratio for yawn detection
    v = get_distance(mesh_points[13], mesh_points[14])
    h = get_distance(mesh_points[78], mesh_points[308])
    return v / h