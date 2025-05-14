from datasets import load_dataset_builder, get_dataset_split_names, load_dataset

# 1. Explore the dataset builder for ESC-50
ds_builder = load_dataset_builder("ashraq/esc50")

# Print dataset description
print("Dataset Description:")
print(ds_builder.info.description)

# Print dataset features
print("\nDataset Features:")
print(ds_builder.info.features)

# Print dataset homepage
print("\nDataset Homepage:")
print(ds_builder.info.homepage)

# 2. See available splits (ESC-50 usually just has 'train')
print("\nAvailable Splits:")
splits = get_dataset_split_names("ashraq/esc50")
print(splits)

# 3. Load a small portion of the dataset to inspect
print("\nLoading dataset...")
try:
    # Corrected line: removed name="main"
    esc50_dataset = load_dataset("ashraq/esc50", split="train")
    print("\nDataset loaded successfully!")
    print(f"Number of examples: {len(esc50_dataset)}")
    print("First example:")
    print(esc50_dataset[0])

    # Let's also check the sampling rate from the first audio file
    if esc50_dataset and 'audio' in esc50_dataset.features:
        first_audio_sample = esc50_dataset[0]['audio']
        if first_audio_sample:
            print(f"\nSampling rate of first audio: {first_audio_sample['sampling_rate']} Hz")
            print(f"Shape of first audio array: {first_audio_sample['array'].shape}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connectivity.")
    print("If the error persists, try changing the load_dataset line to:")
    print("esc50_dataset = load_dataset(\"ashraq/esc50\", name=\"default\", split=\"train\")")