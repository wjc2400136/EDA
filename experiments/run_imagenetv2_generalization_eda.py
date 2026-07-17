from run_imagenet_val_generalization_eda import DatasetDefaults, main


IMAGENETV2_DEFAULTS = DatasetDefaults(
    dataset_name="ImageNet-V2 matched-frequency",
    input_dir="./data_imagenetv2_matched",
    output_dir="./imagenetv2_generalization",
    result_prefix="imagenetv2_matched",
    expected_images=10000,
)


if __name__ == "__main__":
    main(IMAGENETV2_DEFAULTS)
