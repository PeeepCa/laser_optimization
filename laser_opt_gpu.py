import xml.etree.ElementTree as ET
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import euclidean

def nearest_neighbor_algorithm(coords):
    """Apply nearest neighbor algorithm to find an approximate solution to TSP."""
    if len(coords) == 0:
        return []
    
    path = [0]
    used = set(path)
    while len(used) < len(coords):
        last = path[-1]
        next_point = min((i for i in range(len(coords)) if i not in used),
                         key=lambda i: euclidean(coords[last], coords[i]))
        path.append(next_point)
        used.add(next_point)
    return path

def calculate_distances_gpu(coords):
    """Calculate the pairwise Euclidean distance matrix using GPU."""
    if len(coords) == 0:
        return cp.array([])
    
    coords = cp.array(coords)
    diffs = coords[:, cp.newaxis, :] - coords[cp.newaxis, :, :]
    return cp.sqrt(cp.sum(diffs**2, axis=-1))

def path_length_gpu(path, coords_gpu):
    """Calculate the total length of the path on the GPU."""
    if len(path) == 0 or len(coords_gpu) == 0:
        return 0
    
    return cp.sum(cp.sqrt(cp.sum((coords_gpu[path[:-1]] - coords_gpu[path[1:]]) ** 2, axis=1)))

def two_opt_gpu_thread(start, end, path, coords_gpu, best_distance):
    """Perform 2-opt swaps in the given range on the GPU."""
    best_path = path.copy()
    for i in range(start, end):
        for j in range(i + 2, len(path)):  # Ensure valid slices for the swap
            if j <= i + 1:
                continue
            
            # Perform 2-opt swap
            new_path = path.copy()
            new_path[i:j] = path[i:j][::-1]

            # Calculate the length of the new path on GPU
            new_distance = path_length_gpu(new_path, coords_gpu)

            # Check if this path is better
            if new_distance < best_distance:
                best_path = new_path
                best_distance = new_distance
    return best_path, best_distance

def two_opt_multithread_gpu(path, coords_gpu, num_threads=4):
    """Apply 2-opt algorithm using multithreading and GPU."""
    if len(path) == 0 or len(coords_gpu) == 0:
        return path, 0
    
    path = cp.array(path)
    best_distance = path_length_gpu(path, coords_gpu)
    best_path = path.copy()

    improved = True
    while improved:
        improved = False

        # Split the task across multiple threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            step = max(1, len(path) // num_threads)
            futures = []
            for i in range(num_threads):
                start = i * step
                end = min((i + 1) * step, len(path) - 1)
                futures.append(executor.submit(two_opt_gpu_thread, start, end, best_path, coords_gpu, best_distance))

            # Collect results from threads
            for future in futures:
                thread_best_path, thread_best_distance = future.result()
                if thread_best_distance < best_distance:
                    best_path = thread_best_path
                    best_distance = thread_best_distance
                    improved = True

        path = best_path

    return cp.asnumpy(path), best_distance.get()  # Convert back to numpy array and get the distance

def optimize_laser_path_with_multithread_gpu(xml_input_path, xml_output_path):
    # Load the XML file
    tree = ET.parse(xml_input_path)
    root = tree.getroot()

    # Extract coordinates and corresponding blocks from <ID> to <INVERTBADMARK>
    coordinates = []
    blocks = []
    for elem in root.findall('.//TOP_SEQ_LIST'):
        for subelem in elem.findall('TOP_SEQUENCE'):
            xpos = subelem.find('XPOS')
            ypos = subelem.find('YPOS')
            id_elem = subelem.find('ID')
            invert_elem = subelem.find('INVERTBADMARK')

            if xpos is not None and ypos is not None and id_elem is not None and invert_elem is not None:
                try:
                    x = float(xpos.text)
                    y = float(ypos.text)
                    coordinates.append((x, y))
                    blocks.append(subelem)  # Store the whole block starting with <ID> and ending with <INVERTBADMARK>
                except (ValueError, TypeError):
                    continue  # Skip if XPOS or YPOS can't be converted to float

    if len(coordinates) == 0:
        print("No valid coordinates found.")
        return

    # Move coordinates to GPU
    coords_gpu = cp.array(coordinates)

    # Apply the nearest neighbor algorithm
    initial_path = nearest_neighbor_algorithm(coordinates)

    # Improve the solution using 2-opt with GPU and multithreading
    optimized_order, total_distance = two_opt_multithread_gpu(initial_path, coords_gpu)

    # Reorder the blocks according to the optimized path
    optimized_blocks = [blocks[i] for i in optimized_order]

    # Clear the existing elements and append the optimized ones
    for elem in root.findall('.//TOP_SEQ_LIST'):
        elem.clear()
        for block in optimized_blocks:
            elem.append(block)

    # Save the updated XML to the specified output path
    tree.write(xml_output_path)
    print(f"Optimized XML file saved as: {xml_output_path}")
    print(f"Total distance traveled: {total_distance}")

# Example usage:
input_file = 'input.xml'
output_file = 'output.xml'
optimize_laser_path_with_multithread_gpu(input_file, output_file)
