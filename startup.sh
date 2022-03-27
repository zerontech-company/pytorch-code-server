docker run --privileged --rm -it --init \
  --gpus=all \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/projects" \
  --volume="$PWD/config.yml:/home/coder/config/config.yml" \
  -p 8080:8080 \
  -p 3000:3000 \
  -p 43800:43800 \
  zaant-codeserver

  35b27924e9a24c904285e8e6
  033a914589e49e506b42424f