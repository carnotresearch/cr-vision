
import whales

input_shape = (384, 384, 3)
model = whales.get_model(input_shape)
print(model.summary())

whales.compile_model(model)