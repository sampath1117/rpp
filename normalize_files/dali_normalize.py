from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import time

batch_size = 1
axis_mask = 7

def rand_array():
    np.random.seed(0)
    a = np.zeros(24, dtype=np.float32)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                idx = i * 12 + j * 4 + k
                a[idx] = float(idx + 1)
    b = a.reshape(1, 2, 3, 4)
    b = b.view(dtype=np.float32)
    return b

a = rand_array()
print(a)
mean_array = []
std_dev_array = []

#axis mask 0
if axis_mask == 1:
    std_dev_array = np.ones(12, dtype=np.float32)
    mean_array = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(1, 3, 4)
    std_dev_array = std_dev_array.reshape(1, 3, 4)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 2:
    std_dev_array = np.ones(8, dtype=np.float32)
    mean_array = [0, 5, 10, 15, 20, 25, 30, 35]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(2, 1, 4)
    std_dev_array = std_dev_array.reshape(2, 1, 4)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 3:
    std_dev_array = np.ones(4, dtype=np.float32)
    mean_array = [0, 5, 10, 15]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(1, 1, 4)
    std_dev_array = std_dev_array.reshape(1, 1, 4)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 4:
    std_dev_array = np.ones(6, dtype=np.float32)
    mean_array = [0, 5, 10, 15, 20, 25]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(2, 3, 1)
    std_dev_array = std_dev_array.reshape(2, 3, 1)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 5:
    std_dev_array = np.ones(3, dtype=np.float32)
    mean_array = [0, 5, 10]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(1, 3, 1)
    std_dev_array = std_dev_array.reshape(1, 3, 1)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 6:
    std_dev_array = np.ones(2, dtype=np.float32)
    mean_array = [0, 5]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(2, 1, 1)
    std_dev_array = std_dev_array.reshape(2, 1, 1)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)
elif axis_mask == 7:
    std_dev_array = np.ones(1, dtype=np.float32)
    mean_array = [0]
    mean_array = np.array(mean_array, dtype=np.float32)
    mean_array = mean_array.reshape(1, 1, 1)
    std_dev_array = std_dev_array.reshape(1, 1, 1)
    mean_array = mean_array.view(dtype=np.float32)
    std_dev_array = std_dev_array.view(dtype=np.float32)

# print("\nmean: ", mean_array)
# print("stddev: ", std_dev_array)

@pipeline_def
def normalize_pipeline():
    data = fn.external_source(source=rand_array)
    normalized_data = fn.normalize(data, mean = mean_array, stddev = std_dev_array, axes=[0, 1, 2])
    return normalized_data

pipe = normalize_pipeline(batch_size=batch_size, num_threads=1, device_id=None)
pipe.build()
cpu_output = pipe.run()
print("\n\noutput: ")
for i in range(0, batch_size):
    output_data = cpu_output[0].at(i)
    print(output_data)