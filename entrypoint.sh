#!/bin/sh
# export PATH=/home/coder/.local/bin:/home/dong/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
set -eu

# We do this first to ensure sudo works below when renaming the user.
# Otherwise the current container UID may not exist in the passwd database.
eval "$(fixuid -q)"

if [ "${DOCKER_USER-}" ] && [ "$DOCKER_USER" != "$USER" ]; then
  echo "$DOCKER_USER ALL=(ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers.d/nopasswd > /dev/null
  # Unfortunately we cannot change $HOME as we cannot move any bind mounts
  # nor can we bind mount $HOME into a new home as that requires a privileged container.
  sudo usermod --login "$DOCKER_USER" coder
  sudo groupmod -n "$DOCKER_USER" coder

  USER="$DOCKER_USER"

  sudo sed -i "/coder/d" /etc/sudoers.d/nopasswd
fi

rm -rf /projects/main/.aim
rm -rf /projects/.aim
echo "source /home/coder/.bashrc && activate my_env"
# conda activate my_env
cd /projects && aim init && aim up --host 0.0.0.0 &
dagit -d /projects/main -f /projects/main/main.py -h 0.0.0.0 &
dumb-init /usr/bin/code-server --config /home/coder/config/config.yml "$@"

