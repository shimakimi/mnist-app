import numpy
import onnxruntime
from PIL import Image

pil_image = Image.open('sample.png')

# preprocess

resized_image = pil_image.resize((28, 28))
resized_arr = numpy.array(resized_image)
print(f"resized_arr.shape = {resized_arr.shape}")

transposed_arr = resized_arr.transpose(2, 0, 1)
print(f"transposed_arr.shape = {transposed_arr.shape}")

alpha_arr = transposed_arr[3]
print("alpha_arr")
for i in alpha_arr:
    for j in i:
        print(f"%3d " % j, end="")
    print()

reshaped_arr = alpha_arr.reshape(-1)
print( "reshaped_arr")
print(reshaped_arr)

input = [resized_arr.astype(numpy.float32)]
print("input")
print(input)

# predict

onnx_session = onnxruntime.InferenceSession("model.onnx")

output = onnx_session.run(['probabilities'], {'float_input': input})
print("output")
print(output)

result = output[0][0]
print("result")
print(result)