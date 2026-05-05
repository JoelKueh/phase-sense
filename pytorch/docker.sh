podman run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=keep-groups \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ./:/workspace:Z \
  -w /workspace \
  rocm/pytorch
