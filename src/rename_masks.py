import os

mask_folder = "dataset/masks"

for file in os.listdir(mask_folder):

    if "_label" in file:
        new_name = file.replace("_label", "")

        old_path = os.path.join(mask_folder, file)
        new_path = os.path.join(mask_folder, new_name)

        os.rename(old_path, new_path)

print("Renaming complete.")