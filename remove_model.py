import os

input_dir = '/h/u6/c9/01/cheny845/csc413/bitdelta/saved'

# Remove the model file
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file == "ft_compressed.safetensors" or file == "pytorch_model.bin":
            os.remove(os.path.join(root, file))
            print(f"Removed {file}")