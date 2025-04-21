from spine_processing import load_spine_meshes, calculate_sph_harm_metric, save_metrics

if __name__ == "__main__":
    dataset_path = "9009"
    spine_file_pattern = "**/output/**/spine_*.off"
    spine_dataset = load_spine_meshes(dataset_path, spine_file_pattern="**/output/**/spine_*.off")
    metrics_dataset = calculate_sph_harm_metric(spine_dataset, {"l_range": range(15), "sqrt_sample_size": 200})
    save_metrics(metrics_dataset, f"{dataset_path}/spherical_harmonics.csv")
