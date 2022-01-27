import cupy
import pandas
import nvtx

def generate_tensor(shape, dt):
    if dt in [cupy.int8, cupy.uint8, cupy.int16, cupy.uint16, cupy.int32, cupy.uint32, cupy.int64, cupy.uint64]:
        return cupy.random.randint(low=cupy.iinfo(dt).min, high=cupy.iinfo(dt).max, size=shape, dtype=dt)
    elif dt == cupy.float16:
        return cupy.random.normal(size=shape, dtype=cupy.float32).astype(dt)
    elif dt in [cupy.float32, cupy.float64]:
        return cupy.random.normal(size=shape, dtype=dt)
    else:
        raise ValueError(f"Unsupported data type {dt}")

def matmul(shape_a, shape_b, dt):
    rng = nvtx.push_range("generate tensors")
    A = generate_tensor(shape_a, dt)
    B = generate_tensor(shape_b, dt)
    nvtx.pop_range(rng)
    rng = nvtx.push_range("matmul")
    out = cupy.matmul(A, B)
    nvtx.pop_range(rng)
    return out

if __name__ == "__main__":
    inputs = pandas.read_csv("matmul/float32.csv")
    for index, row in inputs.iterrows():
        shape_a = (row["m"], row["k"])
        shape_b = (row["k"], row["n"])
        dt = cupy.dtype(row['dtype'])
        rng = nvtx.push_range(f"{index}")
        matmul(shape_a, shape_b, dt)
        nvtx.pop_range(rng)