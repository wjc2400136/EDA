from run_imagenet_val_generalization_eda import DatasetDefaults, main


IMAGENET_COMPATIBLE_DEFAULTS = DatasetDefaults(
    dataset_name="NIPS 2017 ImageNet-Compatible",
    input_dir="./data",
    output_dir="./imagenet_compatible_generalization",
    result_prefix="imagenet_compatible",
    expected_images=1000,
)


if __name__ == "__main__":
    main(IMAGENET_COMPATIBLE_DEFAULTS)
