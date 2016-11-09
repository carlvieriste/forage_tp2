import pickle
import os.path

n_bytes = 2**31
max_bytes = 2**31 - 1
file_path = 'repr.bin'

print("Reading repr.bin")

bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
count = 0
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)

        count += 1
        print(count)
docVectors = pickle.loads(bytes_in)

count = 0
with open("out.csv", "w") as file:
    for idDoc, doc in docVectors.items():
        file.write(str(idDoc))
        for tf in doc:
            file.write("," + str(tf))
        file.write("\n")

        count += 1
        if count == 50:
            break