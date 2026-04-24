import numpy as np

def array_list():
    #NQ1: Create a 1D NumPy array from the list [10, 20, 30, 40, 50].
    arr = np.array([10, 20, 30, 40, 50])
    return arr

def array_2d():
    #NQ2: Create the following 2D array and print its shape and size.
    arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    return arr2d

def slice_array_2d(arr2d):
    #NQ3: Using the 2D array from Q2, slice out the top-left 2x2 block and print it.
    slice_arr2d = arr2d[0:2, 0:2]
    return slice_arr2d

def zeroes_and_ones():
    #NQ4: Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command.
    arr_zeroes = np.zeros((3, 4))
    arr_ones = np.ones((2, 5))
    return arr_zeroes, arr_ones

def arange_array():
    #NQ5: Create an array using np.arange(0, 50, 5). Then, print the array, its shape, mean, sum, and standard deviation.
    arr_arange=np.arange(0,50,5)
    return arr_arange

def random_normal_array():
    #NQ6: Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()).
    # np.random.normal(loc: Mean, scale: Standard Deviation, size: Output shape)
    # Note: I used np.round() to round the random values to 4 decimal places, without the np.round() the decimal part is long.
    random_arr = np.round(np.random.normal(0, 1, 200), 4)
    return random_arr


def numpy_review():
    print("NQ1:")
    arr_list = array_list()
    print(arr_list)
    print(arr_list.shape, arr_list.dtype, arr_list.ndim)
  
    print("\nNQ2:")
    arr_2d = array_2d()
    print(arr_2d)
    print(arr_2d.shape, arr_2d.size)
   
    print("\nNQ3:")
    print(slice_array_2d(arr_2d))

    print("\nNQ4:")
    arr_zeroes, arr_ones = zeroes_and_ones()
    print(arr_zeroes)
    print(arr_ones)

    print("\nNQ5:")
    arr_arange = arange_array()
    print(arr_arange)
    # Note: I used np.round() to round the random values to 4 decimal places, without the np.round() the decimal part is long.
    print(arr_arange.shape, arr_arange.mean(), arr_arange.sum(), np.round(arr_arange.std(), 2))

    print("\nNQ6:")
    print(random_normal_array())