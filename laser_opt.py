import xml.etree.ElementTree as ET
from scipy.spatial.distance import euclidean
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def nearest_neighbor_algorithm(coords):
    """Apply nearest neighbor algorithm to find an approximate solution to TSP."""
    path = [0]  # start with the first point
    used = set(path)
    while len(used) < len(coords):
        last = path[-1]
        next_point = min((i for i in range(len(coords)) if i not in used),
                         key=lambda i: euclidean(coords[last], coords[i]))
        path.append(next_point)
        used.add(next_point)
    return path

def two_opt_swap(path, i, j):
    """Perform a 2-opt swap."""
    new_path = path[:]
    new_path[i:j] = path[j-1:i-1:-1]
    return new_path

def parallel_two_opt(task):
    """Helper function for multiprocessing: run a single 2-opt step."""
    path, i, j, coords = task
    new_path = two_opt_swap(path, i, j)
    if path_length(new_path, coords) < path_length(path, coords):
        return new_path
    return path

def two_opt(path, coords):
    """Apply 2-opt algorithm to improve the TSP solution using multiprocessing."""
    improved = True
    while improved:
        improved = False
        tasks = [(path, i, j, coords) for i in range(1, len(path) - 2) for j in range(i + 1, len(path)) if j - i > 1]
        
        with Pool(cpu_count()) as pool:
            results = pool.map(parallel_two_opt, tasks)
        
        best_result = min(results, key=lambda p: path_length(p, coords))
        
        if path_length(best_result, coords) < path_length(path, coords):
            path = best_result
            improved = True
    
    return path

def path_length(path, coords):
    """Calculate the total length of the path."""
    return sum(euclidean(coords[path[i]], coords[path[i + 1]]) for i in range(len(path) - 1))

def optimize_laser_path(xml_input_path, xml_output_path):
    # Load the XML file
    tree = ET.parse(xml_input_path)
    root = tree.getroot()

    # Extract XPOS and YPOS coordinates from the XML
    coordinates = []
    for elem in root.findall('.//TOP_SEQ_LIST'):
        for subelem in elem:
            xpos = subelem.find('XPOS')
            ypos = subelem.find('YPOS')
            if xpos is not None and ypos is not None:
                coordinates.append((float(xpos.text), float(ypos.text)))

    # Apply the nearest neighbor algorithm
    initial_path = nearest_neighbor_algorithm(coordinates)

    # Improve the solution using 2-opt with multiprocessing
    optimized_order = two_opt(initial_path, coordinates)

    # Reorder the coordinates according to the optimized path
    optimized_coordinates = [coordinates[i] for i in optimized_order]

    # Update the XML with the optimized coordinates
    for elem, (x, y) in zip(root.findall('.//TOP_SEQ_LIST/*'), optimized_coordinates):
        xpos = elem.find('XPOS')
        ypos = elem.find('YPOS')
        if xpos is not None and ypos is not None:
            xpos.text = str(x)
            ypos.text = str(y)

    # Save the updated XML to the specified output path
    tree.write(xml_output_path)
    print(f"Optimized XML file saved as: {xml_output_path}")

# Example usage:
input_file = 'input.xml'
output_file = 'output.xml'
optimize_laser_path(input_file, output_file)
