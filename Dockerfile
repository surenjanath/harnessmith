# harnessmith: build the harness2 binary, then run the web UI in a slim image.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked && strip target/release/harness2

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates bash python3 git bubblewrap curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 harness \
 && mkdir -p /data && chown harness:harness /data
COPY --from=build /src/target/release/harness2 /usr/local/bin/harness2
USER harness
WORKDIR /home/harness
ENV HARNESS2_RUNS_DIR=/data/runs \
    HARNESS2_CONNECTIONS=/data/connections.json
VOLUME ["/data"]
EXPOSE 8765
# 0.0.0.0 so the port can be published; set HARNESS2_UI_PASSWORD to require a login.
CMD ["harness2", "serve", "--host", "0.0.0.0", "--port", "8765"]
