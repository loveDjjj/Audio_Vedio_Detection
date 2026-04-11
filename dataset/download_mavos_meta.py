# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="unibuc-cs/MAVOS-DD",
#     repo_type="dataset",
#     local_dir="dataset/MAVOS-DD-meta",
#     allow_patterns=[
#         "README.md",
#         "dataset_info.json",
#         "state.json",
#         "data-00000-of-00001.arrow",
#         "dataset.py",
#         "metadata_generation.py",
#     ],
#   )
# print("done")


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unibuc-cs/MAVOS-DD",
    repo_type="dataset",
    local_dir="dataset/MAVOS-DD-english",
    endpoint="https://hf-mirror.com",
    allow_patterns=[
        "README.md",
        "dataset_info.json",
        "state.json",
        "data-00000-of-00001.arrow",
        "dataset.py",
        "metadata_generation.py",
        "english/**",
    ],
    max_workers=1,
)
print("done")