# Minimal sandbox image for the code_executor tool.
#
# Runtime isolation -- no network, read-only root filesystem, memory/CPU caps --
# is enforced by docker-py's container.run() call in src/tools/code_executor.py,
# not by this Dockerfile. This file's only job is a minimal, non-root image with
# nothing beyond the Python standard library installed.

FROM python:3.10-slim

# Code runs as a non-root, non-login user even though the container itself is
# already fully isolated (no network, read-only fs) -- defense in depth.
RUN useradd --no-create-home --uid 1000 --shell /usr/sbin/nologin sandboxuser

WORKDIR /
USER sandboxuser

ENTRYPOINT ["python", "/code.py"]
