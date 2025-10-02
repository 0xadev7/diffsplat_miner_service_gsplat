module.exports = {
  apps: [
    {
      name: "three-gen-generation",
      script: "python",
      args: "-m app.server",
      interpreter: "none",
      env: {
        "HF_HOME": process.env.HF_HOME || "~/.cache/huggingface",
        "TORCH_HOME": process.env.TORCH_HOME || "~/.cache/torch",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1"
      },
      watch: false,
      autorestart: true,
      max_memory_restart: "8G"
    }
  ]
}