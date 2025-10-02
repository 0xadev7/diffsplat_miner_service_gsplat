from huggingface_hub import snapshot_download

def main():
    try:
        snapshot_download("chenguolin/gsdiff_gobj83k_sd15__render", allow_patterns="*")
    except Exception:
        pass
    try:
        snapshot_download("chenguolin/gsdiff_gobj83k_pas_fp16__render", allow_patterns="*")
    except Exception:
        pass
    try:
        snapshot_download("openai/clip-vit-large-patch14")
    except Exception:
        pass

if __name__ == "__main__":
    main()