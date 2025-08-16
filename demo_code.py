
def inefficient_function(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] > data[j]:
                result.append(data[i])
    return result

def long_function_with_many_parameters(a, b, c, d, e, f, g, h):
    # This function is too long and has too many parameters
    x = a + b
    y = c + d
    z = e + f
    w = g + h
    return x + y + z + w
